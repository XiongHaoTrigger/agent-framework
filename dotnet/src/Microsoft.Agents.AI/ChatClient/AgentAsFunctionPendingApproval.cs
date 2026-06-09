// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

/// <summary>
/// Stores the pending approval state for a child agent that was invoked as an <see cref="AIFunction"/>
/// via <see cref="AIAgentExtensions.AsAIFunction"/> and surfaced <see cref="ToolApprovalRequestContent"/> items.
/// </summary>
/// <remarks>
/// This type is stored in the parent agent's <see cref="AgentSessionStateBag"/> so that the
/// <see cref="AgentAsFunctionApprovalDelegatingChatClient"/> can surface the child agent's
/// approval requests to the parent agent's HITL pipeline and later resume the child agent
/// when an approval response is received.
/// </remarks>
internal sealed class AgentAsFunctionPendingApproval
{
    /// <summary>
    /// Gets or sets the child agent that requires approval.
    /// </summary>
    public AIAgent? ChildAgent { get; set; }

    /// <summary>
    /// Gets or sets the session used by the child agent.
    /// When <see langword="null"/>, a new session will be created on resume.
    /// </summary>
    public AgentSession? ChildSession { get; set; }

    /// <summary>
    /// Gets or sets the last messages sent to the child agent before the approval was requested.
    /// These messages are used to resume the child agent's execution context.
    /// </summary>
    public List<ChatMessage>? ChildMessages { get; set; }

    /// <summary>
    /// Gets or sets the approval requests surfaced by the child agent.
    /// </summary>
    public List<ToolApprovalRequestContent>? ApprovalRequests { get; set; }

    /// <summary>
    /// Gets or sets the parent agent's tool call ID associated with this pending approval.
    /// </summary>
    public string? ParentToolCallId { get; set; }
}