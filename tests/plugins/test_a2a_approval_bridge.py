"""Native approval queue remains authoritative across the A2A task boundary."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import ExecApprovalPrompt
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a import protocol
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setattr('tools.approval_context._get_approval_timeout', lambda: 20)
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter.tasks.create('task-a', 'context-a', 'peer-a', *adapter._scope_for_agent(None))
    adapter.tasks.set_state('task-a', protocol.STATE_WORKING)
    adapter._add_pending('task-a', 'context-a')
    yield adapter
    for item in approval.list_gateway_approvals('test-a2a-session'):
        approval.resolve_gateway_approval('test-a2a-session', 'deny', request_id=item['request_id'])


def start_wait(adapter, pool):
    ready = threading.Event()
    def notify(*args, **kwargs):
        prompt = ExecApprovalPrompt(
            chat_id='context-a', session_key='test-a2a-session', text='Approve?',
            actions=[('Once', 'once', ''), ('Deny', 'deny', '')],
            command='print(12 + 18 + 30)', description='read-only sum', smart_denied=False)
        result = adapter._task_approvals.bind(prompt)
        assert result.success, result.error
        ready.set()
    future = pool.submit(_await_gateway_decision, 'test-a2a-session', notify,
                         {'command': 'print(12 + 18 + 30)', 'description': 'read-only sum', 'pattern_key': 'test-sum'})
    assert ready.wait(5), 'native queue did not publish an approval'
    return future


def params(adapter):
    result = adapter._rpc_approvals_list('list', {'taskId': 'task-a', 'contextId': 'context-a'}, 'peer-a')
    request = result['result']['approvals'][0]
    assert request['choices'] == ['once', 'deny']
    assert 'session_key' not in request
    return {'taskId': 'task-a', 'contextId': 'context-a', 'requestId': request['requestId'], 'choice': 'once'}


@pytest.mark.parametrize('choice', ['once', 'deny'])
def test_native_wait_is_resolved_once_without_wider_grant(bridge, choice):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = start_wait(bridge, pool)
        request = params(bridge)
        request['choice'] = choice
        response = bridge._rpc_approval_respond('reply', request, 'peer-a')
        assert response['result']['resolved'] is True
        assert future.result(timeout=5)['choice'] == choice
        assert approval.list_gateway_approvals('test-a2a-session') == []
        assert not approval.is_approval_bypass_active_for_session('test-a2a-session')
        assert 'error' in bridge._rpc_approval_respond('replay', request, 'peer-a')


@pytest.mark.parametrize('attack', ['peer', 'context', 'task', 'scope', 'session', 'always', 'expired', 'withdrawn'])
def test_invalid_reply_cannot_grant_native_approval(bridge, attack):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = start_wait(bridge, pool)
        request = params(bridge)
        peer = 'other-peer' if attack == 'peer' else 'peer-a'
        agent = {'slug': 'other-agent', 'tenant': 'other-tenant'} if attack == 'scope' else None
        changes = {'context': {'contextId': 'other-context'}, 'task': {'taskId': 'other-task'},
                   'session': {'choice': 'session'}, 'always': {'choice': 'always'}}
        request.update(changes.get(attack, {}))
        mutations = {
            'expired': lambda: bridge._task_approvals._bindings[request['requestId']].update(deadline=0),
            'withdrawn': lambda: approval.withdraw_gateway_approval('test-a2a-session', request['requestId'], 'test withdrew'),
        }
        mutate = mutations.get(attack)
        if mutate:
            mutate()
        response = bridge._rpc_approval_respond('bad-reply', request, peer, agent=agent)
        assert 'error' in response
        if attack != 'withdrawn':
            assert len(approval.list_gateway_approvals('test-a2a-session')) == 1
            approval.resolve_gateway_approval('test-a2a-session', 'deny', request_id=request['requestId'])
        decision = future.result(timeout=5)
        assert decision['choice'] != 'once'
        assert not approval.is_approval_bypass_active_for_session('test-a2a-session')


@pytest.mark.parametrize("choice", ["once", "deny"])
def test_nonblocking_send_waits_for_exact_approval_and_returns_final_result(choice):
    async def scenario():
        adapter = A2AAdapter(PlatformConfig(enabled=True))
        adapter._loop = asyncio.get_running_loop()
        adapter._message_handler = object()
        published = asyncio.Event()
        executed = []
        def wait_for_permission(event):
            def notify(*args, **kwargs):
                prompt = ExecApprovalPrompt(
                    chat_id=event.source.chat_id, session_key='nonblocking-a2a-test', text='Approve?',
                    actions=[('Once', 'once', ''), ('Deny', 'deny', '')],
                    command='print(12 + 18 + 30)', description='sum', smart_denied=False)
                assert adapter._task_approvals.bind(prompt).success
                adapter._loop.call_soon_threadsafe(published.set)
            return _await_gateway_decision('nonblocking-a2a-test', notify,
                                          {'command': 'print(12 + 18 + 30)', 'description': 'sum'})
        async def handle(event):
            decision = await asyncio.to_thread(wait_for_permission, event)
            if decision['choice'] == 'once':
                executed.append(12 + 18 + 30)
                await adapter.send(event.source.chat_id, str(executed[-1]), metadata={"notify": True})
            else:
                await adapter.send(event.source.chat_id, "Denied without execution", metadata={"notify": True})
        adapter.handle_message = handle
        request = {'message': protocol.text_message(protocol.ROLE_USER, 'calculate sum', context_id='nb-context'),
                   'configuration': {'blocking': False}}
        sent = await asyncio.wait_for(asyncio.to_thread(adapter._rpc_message_send, 'send', request, 'peer-a',
                                                       v1_response=True), timeout=5)
        task = sent['result']['task']
        assert task['id']
        assert task['status']['state'] == protocol.STATE_WORKING
        await asyncio.wait_for(published.wait(), timeout=5)
        assert executed == []
        query = {'taskId': task['id'], 'contextId': 'nb-context'}
        listed = adapter._rpc_approvals_list('list', query, 'peer-a')['result']['approvals']
        completion = adapter.tasks.watch(task['id'])
        response = adapter._rpc_approval_respond('approve', {**query, 'requestId': listed[0]['requestId'], 'choice': choice}, 'peer-a')
        assert response['result']['resolved']
        await asyncio.wait_for(asyncio.wrap_future(completion), timeout=5)
        final = adapter._rpc_tasks_get('get', {'id': task['id']}, peer='peer-a')['result']
        assert final['status']['state'] == protocol.STATE_COMPLETED
        assert executed == ([60] if choice == 'once' else [])
        assert ('60' if choice == 'once' else 'Denied without execution') in str(final)
    asyncio.run(scenario())
