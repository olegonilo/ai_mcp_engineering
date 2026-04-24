from agents import Runner, trace, gen_trace_id
from clarifier_agent import clarifier_agent, ClarifyingQuestions
from manager_agent import manager_agent


class ResearchManager:

    async def clarify(self, query: str) -> ClarifyingQuestions:
        """Generate 3 clarifying questions for the research query."""
        result = await Runner.run(clarifier_agent, f"Research query: {query}")
        return result.final_output_as(ClarifyingQuestions)

    async def run(self, query: str, clarifications: str = ""):
        """Run the full research pipeline, yielding status updates."""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield "Starting research pipeline..."

            full_input = (
                f"Research query: {query}\n"
                f"User clarifications: {clarifications or 'None provided'}"
            )
            try:
                await Runner.run(manager_agent, full_input, max_turns=25)
                yield "Research complete — report sent to your email."
            except Exception as e:
                yield f"Research failed: {e}"
