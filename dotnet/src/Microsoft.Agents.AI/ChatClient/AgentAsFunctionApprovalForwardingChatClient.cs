// Copyright (c) Microsoft. All rights reserved.

using System;
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
        var parentInvocationContext = FunctionInvokingChatClient.CurrentContext;;

        // 存在审批响应
        if (toolApprovalResponses.Count > 0)
        {
            // 检查输入中是否包含对子 Agent 审批的回复
            Dictionary<string, AgentFunctionContinuationState>? continuations = null;
            var isSucceed = AIAgent.CurrentRunContext?.Session?.StateBag
                .TryGetValue<Dictionary<string, AgentFunctionContinuationState>>(
                    AgentFunctionContinuationState.StateBagKey, out continuations,
                    AgentJsonUtilities.DefaultOptions);

            if (isSucceed == true && continuations is not null)
            {
                // 分组，对于每个agent的多个审批请求，实现一次执行
                // 此时 用户返回的 toolApprovalResponses 中可能有对多个agent的toolApprovalResponse，此处需要以agent为中心去查询每个agent的toolApprovalResponses，即以
                foreach (var continuation in continuations)
                {
                    // 找到这个agent对应的全部审批响应
                    var matchedResponses = toolApprovalResponses
                        .Where(response =>
                            continuation.Value.PendingToolApprovalRequests.Any(request => request.RequestId == response.RequestId))
                        .ToList();
                    if (matchedResponses.Count == 0)
                    {
                        continue;
                    }

                    // 恢复子agent的执行
                    var subAgentToolApprovalResponsesChatMessage = new ChatMessage(ChatRole.User, [..matchedResponses]);
                    var subAgentSession = await continuation.Value.SubAgent.DeserializeSessionAsync(
                            continuation.Value.SubAgentSerializedSession,
                            AgentJsonUtilities.DefaultOptions,
                            cancellationToken)
                        .ConfigureAwait(false);
                    var subAgentResponse = await continuation.Value.SubAgent.RunAsync(
                            subAgentToolApprovalResponsesChatMessage,
                            subAgentSession,
                            cancellationToken: cancellationToken)
                        .ConfigureAwait(false);

                    var nextToolApprovalRequests = subAgentResponse.Messages
                        .SelectMany(m => m.Contents)
                        .OfType<ToolApprovalRequestContent>().ToList();

                    // 删除父agent中的子agent请求响应
                    // 未实现
                    if (nextToolApprovalRequests.Count > 0)
                    {
                        var serializedSession =
                            await continuation.Value.SubAgent.SerializeSessionAsync(
                                subAgentSession,
                                cancellationToken: cancellationToken).ConfigureAwait(false);
                        continuations[continuation.Key] = continuation.Value with
                        {
                            SubAgentSerializedSession = serializedSession,
                            PendingToolApprovalRequests = nextToolApprovalRequests,
                        };

                        return new ChatResponse(new ChatMessage(ChatRole.Assistant, [..nextToolApprovalRequests]));
                    }

                    // 闭环 FRC
                    var closeLoopFunctionResultContext = new FunctionResultContent(
                        continuation.Value.ParentCallSubAgentCallId,
                        subAgentResponse.Text);
                    messages = messages.Append(new ChatMessage(ChatRole.Tool, [closeLoopFunctionResultContext]));

                    // 清理对应agent的StateBag数据
                    continuations.Remove(continuation.Key);
                }

                if (continuations.Count == 0)
                {
                    parentAgentSession?.StateBag.TryRemoveValue(AgentFunctionContinuationState.StateBagKey);
                }
            }
        }

        var response = await base.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);

        parentAgentSession = AIAgent.CurrentRunContext?.Session;
        if (parentAgentSession is null)
        {
            return response;
        }

        // 检查最后一个RFC
        for (int i = response.Messages.Count - 1; i >= 0; i--)
        {
            if (response.Messages[i].Role == ChatRole.Tool)
            {
                var frc = response.Messages[i].Contents.OfType<FunctionResultContent>().FirstOrDefault();
                TryGetContinuationId(frc?.Result, out var continuationId);

                // 查询StateBag
                parentAgentSession.StateBag.TryGetValue<Dictionary<string, AgentFunctionContinuationState>>(
                    AgentFunctionContinuationState.StateBagKey,
                    out var continuations,
                    AgentJsonUtilities.DefaultOptions);

                if (continuations is not null && continuations.TryGetValue(continuationId, out var continuation))
                {
                    // 替换FRC 为 ToolApprovalRequest
                    var toolApprovalRequests = continuation.PendingToolApprovalRequests;
                    response.Messages[i].Role = ChatRole.Assistant;
                    response.Messages[i].Contents = [..toolApprovalRequests];
                }
            }
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

    // private static void ForwardApprovalRequests(IList<ChatMessage> messages, AgentFunctionContinuationState continuation)
    // {
    //     for (int i = messages.Count - 1; i >= 0; i--)
    //     {
    //         var message = messages[i];
    //         if (message.Role != ChatRole.Tool)
    //         {
    //             continue;
    //         }
    //
    //         var res = message.Contents.OfType<FunctionResultContent>().ToList();
    //
    //         for (int j = res.Count - 1; j >= 0; j--)
    //         {
    //             if (TryGetContinuationId(res[j].Result, out var continuationId) &&
    //                 continuationId == continuation.Id)
    //             {
    //                 // replace
    //                 messages.RemoveAt(i);
    //                 var chatMessage = new ChatMessage()
    //                 {
    //                     Role = ChatRole.Assistant,
    //                     Contents = [..continuation.PendingToolApprovalRequests],
    //                 };
    //                 messages.Insert(i, chatMessage);
    //             }
    //         }
    //     }
    // }
}
