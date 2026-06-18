// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

/// <summary>
/// Stores pending approval state for a child agent invoked through <see cref="AIAgentExtensions.AsAIFunction"/>.
/// </summary>
internal sealed class AgentAsFunctionPendingApproval
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AgentAsFunctionPendingApproval"/> class.
    /// </summary>
    public AgentAsFunctionPendingApproval(
        AIAgent childAgent,
        AgentSession? childSession,
        List<ToolApprovalRequestContent> approvalRequests,
        string parentToolCallId)
    {
        this.ChildAgent = childAgent;
        this.ChildSession = childSession;
        this.ApprovalRequests = approvalRequests;
        this.ParentToolCallId = parentToolCallId;
    }

    /// <summary>
    /// Gets the child agent that paused for approval.
    /// </summary>
    public AIAgent ChildAgent { get; }

    /// <summary>
    /// Gets the session used by the child agent.
    /// </summary>
    public AgentSession? ChildSession { get; }

    /// <summary>
    /// Gets or sets the approval requests currently surfaced from the child agent.
    /// </summary>
    public List<ToolApprovalRequestContent> ApprovalRequests { get; set; }

    /// <summary>
    /// Gets the parent agent tool call ID associated with the child invocation.
    /// </summary>
    public string ParentToolCallId { get; }
}
