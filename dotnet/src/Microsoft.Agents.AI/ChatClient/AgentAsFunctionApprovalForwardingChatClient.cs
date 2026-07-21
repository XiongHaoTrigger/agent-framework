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
        var chatMessages = messages as ChatMessage[] ?? messages.ToArray();
        var toolApprovalResponses = chatMessages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalResponseContent>()
            .ToList();

        // 获取当前agent session
        var parentAgentSession = AIAgent.CurrentRunContext?.Session;

        if (toolApprovalResponses.Count > 0)
        {
            // 检查输入中是否包含对子 Agent 审批的回复
            Dictionary<string, AgentFunctionContinuationState>? continuations1 = null;
            var isSucceed = AIAgent.CurrentRunContext?.Session?.StateBag.TryGetValue(
                AgentFunctionContinuationState.StateBagKey, out continuations1, AgentJsonUtilities.DefaultOptions);

            if (isSucceed == true && continuations1 is not null)
            {
                foreach (var toolApprovalResponse in toolApprovalResponses)
                {
                    // 在所有 continuation 的 PendingToolApprovalRequests 中找匹配的 RequestId
                    var matchedContinuation = continuations1.Values
                        .FirstOrDefault(c => c.PendingToolApprovalRequests
                            .Any(r => r.RequestId == toolApprovalResponse.RequestId));

                    if (matchedContinuation is not null)
                    {
                        var subSession = await matchedContinuation.SubAgent
                            .DeserializeSessionAsync(matchedContinuation.SubAgentSerializedSession,
                                cancellationToken: cancellationToken)
                            .ConfigureAwait(false);

                        var approvalMessage = new ChatMessage(ChatRole.User, [toolApprovalResponse]);

                        // Run sub agent
                        var subResponse = await matchedContinuation.SubAgent
                            .RunAsync([approvalMessage], subSession, cancellationToken: cancellationToken)
                            .ConfigureAwait(false);

                        // 查询最后是否存在未审批的请求
                        for (int i = subResponse.Messages.Count - 1; i >= 0; i--)
                        {
                            if (subResponse.Messages[i].Role == ChatRole.User &&
                                subResponse.Messages[i].Contents.Any(c => c is ToolApprovalResponseContent))
                            {
                                break;
                            }

                            if (subResponse.Messages[i].Role == ChatRole.Assistant &&
                                subResponse.Messages[i].Contents.Any(c => c is ToolApprovalRequestContent))
                            {
                                var toolApprovalRequests = subResponse.Messages[i].Contents
                                    .OfType<ToolApprovalRequestContent>()
                                    .ToList();
                                matchedContinuation.PendingToolApprovalRequests = toolApprovalRequests;

                                // 更改父 agent 的 messages
                                var message = new ChatMessage()
                                {
                                    Role = ChatRole.Assistant,
                                    Contents = [..toolApprovalRequests]
                                };
                                return new ChatResponse(chatMessages.Append(message).ToList());
                            }
                        }
                    }
                }

                // 删除处理完成之后的SetBag的数据
                parentAgentSession?.StateBag.TryRemoveValue(AgentFunctionContinuationState.StateBagKey);
            }
        }

        var response = await base.GetResponseAsync(chatMessages, options, cancellationToken).ConfigureAwait(false);

        parentAgentSession = AIAgent.CurrentRunContext?.Session;
        if (parentAgentSession is null)
        {
            return response;
        }

        if (!parentAgentSession.StateBag.TryGetValue<Dictionary<string, AgentFunctionContinuationState>>(
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
                    // replace
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
