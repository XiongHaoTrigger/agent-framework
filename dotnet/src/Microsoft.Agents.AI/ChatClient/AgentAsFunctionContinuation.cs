// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Concurrent;
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
}
