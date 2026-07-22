// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Agents.AI.ChatClient;
using Microsoft.Extensions.AI;
using Microsoft.Shared.Diagnostics;

namespace Microsoft.Agents.AI;

/// <summary>
/// Provides extensions for <see cref="AIAgent"/>.
/// </summary>
public static partial class AIAgentExtensions
{
    /// <summary>
    /// Creates a new <see cref="AIAgentBuilder"/> using the specified agent as the foundation for the builder pipeline.
    /// </summary>
    /// <param name="innerAgent">The <see cref="AIAgent"/> instance to use as the inner agent.</param>
    /// <returns>A new <see cref="AIAgentBuilder"/> instance configured with the specified inner agent.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="innerAgent"/> is <see langword="null"/>.</exception>
    /// <remarks>
    /// This method provides a convenient way to convert an existing <see cref="AIAgent"/> instance into
    /// a builder pattern, enabling easily wrapping the agent in layers of additional functionality.
    /// It is functionally equivalent to using the <see cref="AIAgentBuilder(AIAgent)"/> constructor directly,
    /// but provides a more fluent API when working with existing agent instances.
    /// </remarks>
    public static AIAgentBuilder AsBuilder(this AIAgent innerAgent)
    {
        _ = Throw.IfNull(innerAgent);

        return new AIAgentBuilder(innerAgent);
    }

    /// <summary>
    /// Creates an <see cref="AIFunction"/> that runs the provided <see cref="AIAgent"/>.
    /// </summary>
    /// <param name="agent">The <see cref="AIAgent"/> to be represented as an invocable function.</param>
    /// <param name="options">
    /// Optional metadata to customize the function representation, such as name and description.
    /// If not provided, defaults will be inferred from the agent's properties.
    /// </param>
    /// <param name="session">
    /// Optional <see cref="AgentSession"/> to use for function invocations. If not provided, a new session
    /// will be created for each function call, which may not preserve conversation context.
    /// </param>
    /// <returns>
    /// An <see cref="AIFunction"/> that can be used as a tool by other agents or AI models to invoke this agent.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="agent"/> is <see langword="null"/>.</exception>
    /// <remarks>
    /// <para>
    /// This extension method enables agents to participate in function calling scenarios, where they can be
    /// invoked as tools by other agents or AI models. The resulting function accepts a query string as input and
    /// returns the agent's response as a string, making it compatible with standard function calling interfaces
    /// used by AI models.
    /// </para>
    /// <para>
    /// The resulting <see cref="AIFunction"/> is stateful, referencing both the <paramref name="agent"/> and the optional
    /// <paramref name="session"/>. Especially if a specific session is provided, avoid using the resulting function concurrently
    /// in multiple conversations or in requests where the parallel function calls may result in concurrent usage of the session,
    /// as that could lead to undefined and unpredictable behavior.
    /// </para>
    /// </remarks>
    public static AIFunction AsAIFunction(this AIAgent agent, AIFunctionFactoryOptions? options = null,
        AgentSession? session = null)
    {
        Throw.IfNull(agent);

        [Description("Invoke an agent to retrieve some information.")]
        async Task<object> InvokeAgentAsync(
            [Description("Input query to invoke the agent.")]
            string query,
            CancellationToken cancellationToken)
        {
            // Propagate any additional properties from the parent agent's run to the child agent if the parent is using a FunctionInvokingChatClient.
            AgentRunOptions? agentRunOptions =
                FunctionInvokingChatClient.CurrentContext?.Options?.AdditionalProperties is
                    AdditionalPropertiesDictionary dict
                    ? new AgentRunOptions { AdditionalProperties = dict }
                    : null;

            // Save base agent data
            var parentSession = AIAgent.CurrentRunContext?.Session;
            var parentInvocationContext = FunctionInvokingChatClient.CurrentContext;

            // Get sub agent session
            var subSession = session ?? await agent.CreateSessionAsync(cancellationToken).ConfigureAwait(false);

            // run sub agent
            var response = await agent
                .RunAsync(query, session: subSession, options: agentRunOptions, cancellationToken: cancellationToken)
                .ConfigureAwait(false);

            // Get tool approval requests
            var toolApprovalRequests = response.Messages
                .SelectMany(m => m.Contents)
                .OfType<ToolApprovalRequestContent>().ToList();

            // Need to handle tool approval requests if there are any approval requests
            if (toolApprovalRequests.Count > 0)
            {
                if (parentSession is null)
                {
                    throw new InvalidOperationException("A parent session is required to resume the child agent.");
                }

                if (parentInvocationContext?.CallContent is not { } parentFunctionCallContext)
                {
                    throw new InvalidOperationException(
                        "The parent function call context is unavailable.");
                }

                var serializedSession = await agent
                    .SerializeSessionAsync(subSession, cancellationToken: cancellationToken).ConfigureAwait(false);

                var continuation = new AgentFunctionContinuationState
                {
                    ContinuationId = Guid.NewGuid().ToString("N"),
                    SubAgent = agent,
                    ParentCallSubAgentCallName = parentFunctionCallContext.Name,
                    SubAgentSerializedSession = serializedSession,
                    PendingToolApprovalRequests = toolApprovalRequests,
                    ParentCallSubAgentCallId = parentFunctionCallContext.CallId,
                };
                var continuations =
                    parentSession.StateBag.TryGetValue<Dictionary<string, AgentFunctionContinuationState>>(
                        AgentFunctionContinuationState.StateBagKey,
                        out Dictionary<string, AgentFunctionContinuationState>? storedContinuation,
                        AgentJsonUtilities.DefaultOptions) && storedContinuation is not null
                        ? storedContinuation
                        : new Dictionary<string, AgentFunctionContinuationState>();

                continuations[continuation.ContinuationId] = continuation;

                // Save approval requests of sub agent to state bag of parent agent
                parentSession.StateBag.SetValue(AgentFunctionContinuationState.StateBagKey, continuations,
                    AgentJsonUtilities.DefaultOptions);

                parentInvocationContext.Terminate = true;

                return new AgentAsFunctionResult { ContinuationId = continuation.ContinuationId };
            }

            return response.Text;
        }

        options ??= new();
        options.Name ??= SanitizeAgentName(agent.Name);
        options.Description ??= agent.Description;

        return AIFunctionFactory.Create(InvokeAgentAsync, options);
    }

    /// <summary>
    /// Removes characters from AI agent name that shouldn't be used in an AI function name.
    /// </summary>
    /// <param name="agentName">The AI agent name to sanitize.</param>
    /// <returns>
    /// The sanitized agent name with invalid characters replaced by underscores, or <c>null</c> if the input is <c>null</c>.
    /// </returns>
    private static string? SanitizeAgentName(string? agentName)
    {
        return agentName is null
            ? agentName
            : InvalidNameCharsRegex().Replace(agentName, "_");
    }

    /// <summary>Regex that flags any character other than ASCII digits or letters.</summary>
#if NET
    [GeneratedRegex("[^0-9A-Za-z]+")]
    private static partial Regex InvalidNameCharsRegex();
#else
    private static Regex InvalidNameCharsRegex() => s_invalidNameCharsRegex;

    private static readonly Regex s_invalidNameCharsRegex = new("[^0-9A-Za-z]+", RegexOptions.Compiled);
#endif
}
