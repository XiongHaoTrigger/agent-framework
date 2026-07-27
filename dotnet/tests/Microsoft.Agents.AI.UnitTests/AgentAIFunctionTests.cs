// Copyright (c) Microsoft. All rights reserved.

using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Moq;

namespace Microsoft.Agents.AI.UnitTests;

public class AgentAIFunctionTests
{
    [Fact]
    public async Task InvokeAsync_DelegatesToInnerFunctionAndExposesAgentAsync()
    {
        // Arrange
        var agent = new Mock<AIAgent>().Object;
        AIFunction innerFunction = AIFunctionFactory.Create(
            (string query) => $"Processed: {query}",
            new AIFunctionFactoryOptions
            {
                Name = "TestAgentFunction",
                Description = "Test description",
            });
        var agentFunction = new AgentAIFunction(agent, innerFunction);

        // Act
        object? result = await agentFunction.InvokeAsync(
            new AIFunctionArguments
            {
                ["query"] = "hello",
            });

        // Assert
        Assert.Same(agent, agentFunction.Agent);
        Assert.Same(agentFunction, agentFunction.GetService<AgentAIFunction>());
        Assert.Equal(innerFunction.Name, agentFunction.Name);
        Assert.Equal(innerFunction.Description, agentFunction.Description);
        Assert.Equal("Processed: hello", result?.ToString());
    }
}
