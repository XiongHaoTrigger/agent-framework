// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.ChatClient;

internal sealed class AgentAsFunctionApprovalForwardingChatClient : DelegatingChatClient
{
    private const string ContinuationPrefix = "__agent_continuation__:";

    public AgentAsFunctionApprovalForwardingChatClient(IChatClient innerClient) : base(innerClient) { }

    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // 必须在调用内部客户端前捕获父 Session；内部调用可能切换 CurrentRunContext。
        var parentSession = AIAgent.CurrentRunContext?.Session;

        // 如果消息中包含当前待处理的审批响应，先恢复对应的子 Agent。
        var messageList = messages.ToList();
        if (parentSession is not null)
        {
            var resumeResult = await this.TryResumeChildAgentAsync(messageList, options, parentSession, cancellationToken)
                .ConfigureAwait(false);
            if (resumeResult?.ApprovalResponse is not null)
            {
                // 子 Agent 再次请求审批，不能继续父 Agent。
                return resumeResult.ApprovalResponse;
            }

            if (resumeResult?.ParentMessages is not null)
            {
                // 子 Agent 已完成，继续父 Agent。
                messageList = resumeResult.ParentMessages;
            }
        }

        var response = await base.GetResponseAsync(messageList, options, cancellationToken).ConfigureAwait(false);

        if (parentSession is null)
        {
            return response;
        }

        var isSucceed = parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out var continuations,
            AgentJsonUtilities.DefaultOptions);

        if (isSucceed == false || continuations is null)
        {
            return response;
        }

        foreach (var message in response.Messages)
        {
            List<AIContent> replaceContents = [];
            for (int i = 0; i < message.Contents.Count; i++)
            {
                var content = message.Contents[i];
                if (TryGetContinuationId(content, out var continuationId) &&
                    continuations.TryGetValue(continuationId, out var continuation))
                {
                    replaceContents.AddRange(continuation.PendingToolApprovalRequestDict.Values);
                }
                else
                {
                    replaceContents.Add(content);
                }
            }
            message.Contents = replaceContents;
        }
        return response;
    }

    private static bool TryGetContinuationId(
        AIContent content,
        out string continuationId)
    {
        continuationId = string.Empty;

        if (content is not FunctionResultContent { Result: string result } ||
            !result.StartsWith(ContinuationPrefix, StringComparison.Ordinal))
        {
            return false;
        }

        continuationId = result[ContinuationPrefix.Length..];
        return continuationId.Length > 0;
    }

    private async Task<ChildResumeResult?> TryResumeChildAgentAsync(List<ChatMessage> messages, ChatOptions? options,
        AgentSession parentSession, CancellationToken cancellationToken)
    {
        // continuation 保存了正在等待的审批请求以及对应的子 Agent Session。
        if (!parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
                AgentAsFunctionContinuation.StateBagKey,
                out var continuations,
                AgentJsonUtilities.DefaultOptions) || continuations is null)
        {
            return null;
        }

        // 消息历史里可能同时存在旧审批响应和父 Agent 自己的审批响应。
        // 这里只选择 RequestId 与当前 pending 请求匹配的那一个。
        var approvalResponse = messages
            .SelectMany(message => message.Contents)
            .OfType<ToolApprovalResponseContent>()
            .FirstOrDefault(response =>
                continuations.Values.Any(continuation =>
                    continuation.PendingToolApprovalRequestDict.ContainsKey(response.RequestId)));

        if (approvalResponse is null)
        {
            return null;
        }

        var continuationEntry = continuations.First(pair =>
            pair.Value.PendingToolApprovalRequestDict.ContainsKey(approvalResponse.RequestId));
        var continuationId = continuationEntry.Key;
        var continuation = continuationEntry.Value;
        var originalRequest =
            continuation.PendingToolApprovalRequestDict[approvalResponse.RequestId];

        // StateBag 必须可序列化，不能直接保存 Agent 引用。
        // 因此使用父函数调用名，从本次可用工具中找回原来的 AgentAIFunction。
        var ficc = this.GetService<FunctionInvokingChatClient>();
        var agentFunction = (options?.Tools ?? Enumerable.Empty<AITool>())
            .Concat(ficc?.AdditionalTools ?? Enumerable.Empty<AITool>())
            .OfType<AIFunction>()
            .Select(tool => tool.GetService<AgentAIFunction>())
            .FirstOrDefault(tool =>
                tool is not null && string.Equals(tool.Name, continuation.ParentCallName, StringComparison.Ordinal));

        if (agentFunction is null)
        {
            throw new InvalidOperationException(
                $"Cannot find agent function '{continuation.ParentCallName}'.");
        }

        // 使用保存的状态恢复原子 Agent 的 Session，而不是创建一个新 Session。
        var childSession = await agentFunction.Agent.DeserializeSessionAsync(
            continuation.SerializedSession,
            AgentJsonUtilities.DefaultOptions,
            cancellationToken).ConfigureAwait(false);

        // 使用原始审批请求中的 ToolCall，不能信任用户传入的 ToolCall。
        var reboundApproval = new ToolApprovalResponseContent(
            approvalResponse.RequestId,
            approvalResponse.Approved,
            originalRequest.ToolCall)
        {
            Reason = approvalResponse.Reason,
        };

        // 把审批结果作为用户消息传回子 Agent，使它从暂停位置继续执行。
        var childResponse = await RunChildAgentAsync(agentFunction.Agent, childSession, reboundApproval, cancellationToken)
            .ConfigureAwait(false);

        var newApprovalRequests = childResponse.Messages
            .SelectMany(message => message.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();

        if (newApprovalRequests.Count > 0)
        {
            // 子 Agent 再次请求审批：保存最新 Session，并用新请求替换当前 pending 请求。
            var serializedSession = await agentFunction.Agent
                .SerializeSessionAsync(childSession, AgentJsonUtilities.DefaultOptions, cancellationToken)
                .ConfigureAwait(false);
            var surfacedRequestIds = new HashSet<string>(
                continuation.SurfacedToolApprovalRequestIds,
                StringComparer.Ordinal);
            surfacedRequestIds.UnionWith(newApprovalRequests.Select(request => request.RequestId));

            continuations[continuationId] = continuation with
            {
                SerializedSession = serializedSession,
                PendingToolApprovalRequestDict = new ConcurrentDictionary<string, ToolApprovalRequestContent>(
                    newApprovalRequests.ToDictionary(request => request.RequestId, request => request)),
                SurfacedToolApprovalRequestIds = surfacedRequestIds,
            };

            parentSession.StateBag.SetValue(
                AgentAsFunctionContinuation.StateBagKey,
                continuations,
                AgentJsonUtilities.DefaultOptions);

            var approvalMessages = childResponse.Messages
                .Select(message =>
                {
                    var clone = message.Clone();
                    clone.Contents = message.Contents
                        .OfType<ToolApprovalRequestContent>()
                        .Cast<AIContent>()
                        .ToList();

                    return clone;
                })
                .Where(message => message.Contents.Count > 0)
                .ToList();

            return new ChildResumeResult(ApprovalResponse: new ChatResponse(approvalMessages));
        }

        // 子 Agent 已完成，不再需要保存 continuation。
        continuations.Remove(continuationId);

        if (continuations.Count == 0)
        {
            parentSession.StateBag.TryRemoveValue(AgentAsFunctionContinuation.StateBagKey);
        }
        else
        {
            parentSession.StateBag.SetValue(AgentAsFunctionContinuation.StateBagKey, continuations, AgentJsonUtilities.DefaultOptions);
        }

        // 消息历史会包含前几轮审批。最终回到父 Agent 前，要一次性删除属于该子 Agent 的全部审批消息。
        // 同时合并当前 pending ID，以兼容没有累计字段的旧 continuation。
        var surfacedRequestIdsToRemove = new HashSet<string>(
            continuation.SurfacedToolApprovalRequestIds,
            StringComparer.Ordinal);
        surfacedRequestIdsToRemove.UnionWith(continuation.PendingToolApprovalRequestDict.Keys);
        var rewrittenMessages = new List<ChatMessage>();

        foreach (var message in messages)
        {
            var contents = message.Contents
                .Where(content => content switch
                {
                    ToolApprovalRequestContent request =>
                        !surfacedRequestIdsToRemove.Contains(request.RequestId),
                    ToolApprovalResponseContent response =>
                        !surfacedRequestIdsToRemove.Contains(response.RequestId),
                    _ => true,
                })
                .ToList();

            if (contents.Count > 0)
            {
                var clone = message.Clone();
                clone.Contents = contents;
                rewrittenMessages.Add(clone);
            }
        }

        // 子 Agent 的最终文本就是父 Agent 此次函数调用的结果。
        rewrittenMessages.Add(new ChatMessage(
            ChatRole.Tool,
            [
                new FunctionResultContent(
                    continuation.ParentCallId!,
                    childResponse.Text),
            ]));

        return new ChildResumeResult(ParentMessages: rewrittenMessages);
    }

    private static async Task<AgentResponse> RunChildAgentAsync(
        AIAgent agent,
        AgentSession session,
        ToolApprovalResponseContent approvalResponse,
        CancellationToken cancellationToken)
    {
        return await agent.RunAsync(
                [new ChatMessage(ChatRole.User, [approvalResponse])],
                session,
                cancellationToken: cancellationToken)
            .ConfigureAwait(false);
    }

    private sealed record ChildResumeResult(
        List<ChatMessage>? ParentMessages = null,
        ChatResponse? ApprovalResponse = null);
}
