"""Task-scoped bridge to native, single-use gateway approvals.

The native queue remains authoritative. A binding only exposes the exact pending
request rendered by the gateway; neither a session key nor a wider grant crosses A2A.
"""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

from gateway.platforms.base import SendResult
from . import protocol


class TaskApprovals:
    def __init__(self, adapter):
        self.adapter = adapter
        self._bindings: dict[str, dict] = {}
        self._lock = threading.RLock()

    def _owner(self, task_id, context_id, peer, agent):
        rec = self.adapter.tasks.get(task_id)
        scope = self.adapter._scope_for_agent(agent)
        if not rec or rec['peer'] != peer or rec['context_id'] != context_id:
            return None
        if (rec.get('agent_slug', ''), rec.get('tenant', '')) != scope:
            return None
        return rec

    @staticmethod
    def _pending(binding):
        from tools.approval import list_gateway_approvals
        return time.monotonic() < binding['deadline'] and any(
            item.get('request_id') == binding['request_id']
            for item in list_gateway_approvals(binding['session_key']))

    def bind(self, prompt):
        from tools.approval import list_gateway_approvals
        from tools.approval_context import _get_approval_timeout
        from gateway.run import _redact_approval_command
        with self.adapter._pending_lock:
            candidates = [tid for tid, (ctx, future) in self.adapter._pending.items()
                          if ctx == prompt.chat_id and not future.done()]
        if len(candidates) != 1:
            return SendResult(success=False, error='No unique active A2A task for approval')
        task_id = candidates[0]
        with self._lock:
            requests = [item for item in list_gateway_approvals(prompt.session_key)
                        if item.get('request_id') not in self._bindings
                        and _redact_approval_command(item.get('command', '')) == prompt.command]
            if len(requests) != 1:
                return SendResult(success=False, error='No unique native approval request')
            item = requests[0]
            ttl = max(0, _get_approval_timeout())
            rid = item['request_id']
            self._bindings[rid] = {
                'task_id': task_id, 'session_key': prompt.session_key, 'request_id': rid,
                'deadline': time.monotonic() + ttl,
                'public': {'requestId': rid, 'command': prompt.command,
                           'description': prompt.description, 'choices': ['once', 'deny'],
                           'expiresAt': datetime.fromtimestamp(time.time() + ttl, UTC).isoformat()},
            }
        return SendResult(success=True, message_id=rid)

    def list(self, task_id, context_id, peer, agent):
        if not self._owner(task_id, context_id, peer, agent):
            raise ValueError('Task approval scope not found')
        with self._lock:
            return [dict(binding['public']) for binding in self._bindings.values()
                    if binding['task_id'] == task_id and self._pending(binding)]

    def respond(self, task_id, context_id, peer, agent, request_id, choice):
        from tools.approval import resolve_gateway_approval
        if choice not in ('once', 'deny'):
            raise ValueError('Only once or deny is supported')
        rec = self._owner(task_id, context_id, peer, agent)
        if not rec or rec['state'] in protocol.TERMINAL_STATES:
            raise ValueError('Task approval scope not found or task ended')
        with self._lock:
            binding = self._bindings.get(request_id)
            if not binding or binding['task_id'] != task_id or not self._pending(binding):
                raise ValueError('Approval expired, resolved, or not found')
            # Consume before waking the tool; concurrent replies cannot issue a second grant.
            del self._bindings[request_id]
            if resolve_gateway_approval(binding['session_key'], choice, request_id=request_id) != 1:
                raise ValueError('Approval is no longer pending')
        return {'taskId': task_id, 'requestId': request_id, 'resolved': True}

    def discard(self, task_id):
        with self._lock:
            for rid in [rid for rid, binding in self._bindings.items() if binding['task_id'] == task_id]:
                del self._bindings[rid]
