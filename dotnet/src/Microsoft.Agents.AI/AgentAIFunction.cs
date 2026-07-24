// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

internal sealed class AgentAIFunction : DelegatingAIFunction
{
    public AIAgent Agent { get; }

    public AgentAIFunction(AIAgent agent, AIFunction innerFunction) : base(innerFunction)
    {
        this.Agent = agent;
    }
}
