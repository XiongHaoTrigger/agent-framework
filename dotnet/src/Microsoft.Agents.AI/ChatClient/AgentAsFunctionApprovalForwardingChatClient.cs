// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.ChatClient;

internal class AgentAsFunctionApprovalForwardingChatClient(IChatClient innerClient) : DelegatingChatClient(innerClient)
{
    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = new CancellationToken())
    {
        // 1. 检查输入中是否包含对子 Agent 审批的回复
        // 2. 根据 RequestId 找到 continuation
        // 3. 找到对应的子 Agent
        // 4. 反序列化 SubAgentSerializedSession
        // 5. 先恢复并运行子 Agent
        // 6. 子 Agent 完成后，生成父调用对应的 FunctionResultContent
        // 7. 再调用 base.GetResponseAsync，让父 Agent 继续

        var response = await base.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);

        var parentSession = AIAgent.CurrentRunContext?.Session;
        if (parentSession is null)
        {
            return response;
        }

        if (!parentSession.StateBag.TryGetValue<Dictionary<string, AgentFunctionContinuationState>>(
                AgentFunctionContinuationState.StateBagKey, out var continuations, AgentJsonUtilities.DefaultOptions) ||
            continuations is null)
        {
            return response;
        }

        foreach (AgentFunctionContinuationState agentFunctionContinuationState in continuations.Values)
        {
            ForwardApprovalRequests(response.Messages, agentFunctionContinuationState);
        }

        return response;
    }

    public override IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = new CancellationToken())
    {
        return base.GetStreamingResponseAsync(messages, options, cancellationToken);
    }

    private static bool TryGetContinuationId(object? result, out string continuationId)
    {
        // if result is AgentAsFunctionResult, get continuationId from result
        if (result is AgentAsFunctionResult agentAsFunctionResult)
        {
            continuationId = agentAsFunctionResult.ContinuationId;
            return true;
        }

        // if result is JsonElement, get continuationId from result
        if (result is JsonElement { ValueKind: JsonValueKind.Object } jsonElement &&
            jsonElement.TryGetProperty("continuationId", out var continuationIdElement) &&
            continuationIdElement.GetString() is { } id)
        {
            continuationId = id;
            return true;
        }

        // if result is string, return false
        continuationId = string.Empty;
        return false;
    }

    /// <summary>
    /// Replace sub agent FCC with approval requests
    /// </summary>
    /// <param name="messages">Chat Messages</param>
    /// <param name="continuation">Saved sub agent state</param>
    private static void ForwardApprovalRequests(IList<ChatMessage> messages, AgentFunctionContinuationState continuation)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            var message = messages[i];
            if (message.Role != ChatRole.Tool)
            {
                continue;
            }

            var res = message.Contents.OfType<FunctionResultContent>().ToList();

            for (int j = res.Count - 1; j >= 0; j--)
            {
                if (TryGetContinuationId(res[j].Result, out var continuationId) &&
                    continuationId == continuation.Id)
                {
                    // repleace
                    messages.RemoveAt(i);
                    var chatMessage = new ChatMessage()
                    {
                        Role = ChatRole.Assistant,
                        Contents = [..continuation.PendingToolApprovalRequests],
                    };
                    messages.Insert(i, chatMessage);
                }
            }
        }
    }
}
