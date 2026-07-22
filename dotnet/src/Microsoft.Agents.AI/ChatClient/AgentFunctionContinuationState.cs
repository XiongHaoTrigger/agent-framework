// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.ChatClient;

internal sealed record AgentFunctionContinuationState
{
    public const string StateBagKey = "__AgentFunctionContinuationState__";
    public string ContinuationId { get; init; } = string.Empty;

    public AIAgent SubAgent { get; init; } = null!;

    public JsonElement SubAgentSerializedSession { get; init; }

    public List<ToolApprovalRequestContent> PendingToolApprovalRequests { get; set; } = [];

    public string ParentCallSubAgentCallName { get; init; } = string.Empty;

    public string ParentCallSubAgentCallId { get; init; } = string.Empty;
}
