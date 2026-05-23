import asyncio
import os
import time
from typing import Any

import dotenv
from github import Github
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq

dotenv.load_dotenv()

# --- GitHub client setup ---
# In GitHub Actions, GITHUB_TOKEN is provided; locally we read from .env
git_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
git = Github(git_token) if git_token else None

# Repo can come from REPOSITORY (GitHub Actions: "owner/repo")
# or GITHUB_REPO_URL (local: full URL)
repository = os.getenv("REPOSITORY", "")
repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/Nkwachi-N/recipes-api.git")

if repository:
    full_repo_name = repository
    repo = git.get_repo(full_repo_name) if git is not None else None
elif repo_url:
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    username = repo_url.split("/")[-2]
    full_repo_name = f"{username}/{repo_name}"
    repo = git.get_repo(full_repo_name) if git is not None else None
else:
    repo = None

# --- LLM setup ---
if os.getenv("GROQ_API_KEY"):
    llm = Groq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
    )
else:
    llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )




# --- Helper to read/write state across LlamaIndex versions ---
async def _get_state(ctx):
    if hasattr(ctx, "get") and callable(ctx.get):
        try:
            return await ctx.get("state")
        except Exception:
            pass
    if hasattr(ctx, "state") and isinstance(getattr(ctx, "state"), dict):
        return ctx.state
    if hasattr(ctx, "data") and isinstance(getattr(ctx, "data"), dict):
        return ctx.data
    if not hasattr(ctx, "_fallback_state"):
        ctx._fallback_state = {"gathered_contexts": "", "review_comment": "", "final_review": ""}
    return ctx._fallback_state


async def _set_state(ctx, state):
    if hasattr(ctx, "set") and callable(ctx.set):
        try:
            await ctx.set("state", state)
            return
        except Exception:
            pass


# --- State management functions ---

async def add_context_to_state(ctx: Context, context: str) -> str:
    """Useful for adding the gathered context to the state."""
    current_state = await _get_state(ctx)
    current_state["gathered_contexts"] = context
    await _set_state(ctx, current_state)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for adding the draft review comment to the state."""
    current_state = await _get_state(ctx)
    current_state["review_comment"] = draft_comment
    await _set_state(ctx, current_state)
    return "State updated with review comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final review to the state."""
    current_state = await _get_state(ctx)
    current_state["final_review"] = final_review
    await _set_state(ctx, current_state)
    return "State updated with final review."


# --- Tool functions ---

def get_pull_request_details(pr_number: int) -> dict[str, Any]:
    """Fetch details of a pull request given its number. Returns author, title, body,
    diff_url, state, and commit SHAs."""
    pull_request = repo.get_pull(pr_number)

    commit_SHAs = []
    commits = pull_request.get_commits()
    for c in commits:
        commit_SHAs.append(c.sha)

    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commit_SHAs": commit_SHAs,
    }


def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository given its file path."""
    contents = repo.get_contents(file_path)
    return contents.decoded_content.decode("utf-8")


def get_pr_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Fetch details of a commit given its SHA. Returns a list of changed files with
    filename, status, additions, deletions, changes, and patch (diff)."""
    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post a final review comment to a pull request on GitHub. Only call this when you have a complete review comment ready to post."""
    if not comment or not comment.strip():
        return "Error: Cannot post an empty review. Please provide a review comment."
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=comment, event="COMMENT")
    return f"Review comment posted to PR #{pr_number}."


# --- Convert functions to tools ---

pr_details_tool = FunctionTool.from_defaults(get_pull_request_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
pr_commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)

# --- Agents ---

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository including PR details, changed files, and file contents.",
    tools=[pr_details_tool, file_contents_tool, pr_commit_details_tool, add_context_to_state],
    system_prompt=(
        "You are the context gathering agent. When gathering context, you MUST gather \n: "
        "- The details: author, title, body, diff_url, state, and head_sha; \n"
        "- Changed files; \n"
        "- Any requested for files; \n"
        "IMPORTANT: You MUST call tools ONE AT A TIME in this exact order: \n"
        "1. First call get_pull_request_details to get PR info and commit SHAs. \n"
        "2. Wait for the result, then call get_pr_commit_details with the actual SHA string from step 1. \n"
        "3. Then fetch any file contents needed. \n"
        "4. Then save context to state. \n"
        "NEVER call get_pr_commit_details and get_pull_request_details in the same turn. \n"
        "Once you gather the requested info, you MUST hand control back to the Commentor Agent."
    ),
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[add_comment_to_state],
    system_prompt=(
        "You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n "
        "Ensure to do the following for a thorough review: "
        "- Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. "
        "- Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n"
        "    - What is good about the PR? \n"
        "    - Did the author follow ALL contribution rules? What is missing? \n"
        "    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n"
        "    - Are new endpoints documented? - use the diff to determine this. \n "
        "    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n"
        "- If you need any additional details, you must hand off to the ContextAgent. \n"
        '- You should directly address the author. So your comments should sound like: \n'
        '"Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"\n'
        "CRITICAL WORKFLOW - You MUST follow these steps in order: \n"
        "1. After drafting the review, you MUST call add_comment_to_state with the full review as the draft_comment argument. \n"
        "2. Then you MUST call handoff to transfer control to ReviewAndPostingAgent. \n"
        "DO NOT just write the review as a text response. You MUST use the tools. Failure to call these tools means the review will not be posted."
    ),
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment, checks if it meets quality criteria, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt=(
        "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. \n"
        "Once a review is generated, you need to run a final check and post it to GitHub.\n"
        "   - The review must: \n"
        "   - Be a ~200-300 word review in markdown format. \n"
        "   - Specify what is good about the PR: \n"
        "   - Did the author follow ALL contribution rules? What is missing? \n"
        "   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n"
        "   - Are there notes on whether new endpoints were documented? \n"
        "   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n"
        " If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n"
        " When you are satisfied, save the final review to state and post it to GitHub.\n"
        " IMPORTANT: Do NOT call post_review_to_github until you have received a complete review from the CommentorAgent. \n"
        " On your first turn, ONLY hand off to the CommentorAgent. Do NOT call any other tools on the first turn. \n"
        " After successfully posting, you are DONE. Do NOT hand off again. Respond with a summary."
    ),
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)


# --- Main ---

async def main():
    pr_number = os.getenv("PR_NUMBER", "1")
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    try:
        handler = workflow_agent.run(prompt.format())

        current_agent = None
        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"Current agent: {current_agent}")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print("\n\nFinal response:", event.response.content)
                if event.tool_calls:
                    print("Selected tools: ", [call.tool_name for call in event.tool_calls])
            elif isinstance(event, ToolCallResult):
                print(f"Output from tool: {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")
            if type(event).__name__ == "WorkflowFailedEvent":
                print(f"WORKFLOW FAILED - exception: {getattr(event, 'exception', 'no exception attr')}")
                print(f"WORKFLOW FAILED - step: {getattr(event, 'step_name', 'no step name')}")

        # Also try to get final result
        try:
            result = await handler
            print(f"Handler result: {result}")
        except Exception as e:
            print(f"Handler exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Outer exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
    if git is not None:
        git.close()