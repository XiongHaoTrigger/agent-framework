// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.ChatClient;

internal sealed record AgentFunctionContinuationState
{
    public string Id { get; } = Guid.NewGuid().ToString("N");

    public const string StateBagKey = "__sub_agent_continuation_state";

    public AIAgent SubAgent { get; init; } = null!;

    public JsonElement SubAgentSerializedSession { get; init; }

    public List<ToolApprovalRequestContent> PendingToolApprovalRequests { get; set; } = [];

    public string AgentAsFunctionName { get; init; } = string.Empty;

    public FunctionCallContent ParentFunctionCallContext { get; init; } = null!;
}
