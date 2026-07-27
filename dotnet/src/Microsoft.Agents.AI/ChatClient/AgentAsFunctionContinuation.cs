// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

internal sealed record AgentAsFunctionContinuation
{
    public const string StateBagKey = "__agentAsFunctionContinuations";
    public string? ParentCallName { get; init; } = string.Empty;

    public string? ParentCallId { get; init; } = string.Empty;

    public JsonElement SerializedSession { get; init; }

    public ConcurrentDictionary<string, ToolApprovalRequestContent> PendingToolApprovalRequestDict { get; init; } = [];

    // 累计记录已经向父 Agent 冒泡的审批请求，子 Agent 最终完成时需要从父消息历史中统一移除。
    public HashSet<string> SurfacedToolApprovalRequestIds { get; init; } = [];
}
