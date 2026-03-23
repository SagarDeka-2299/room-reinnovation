from langchain_core.messages.content import create_text_block, create_image_block, ImageContentBlock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, cast
from PIL import Image
import io
import os
import base64
from pathlib import Path
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
load_dotenv()


def local_image_message_content(path:Path)->ImageContentBlock:
    return create_image_block(
        base64=base64.b64encode(path.read_bytes()).decode("utf-8"),
        mime_type=f"image/{path.suffix.lstrip('.')}"
    )

class Plans(BaseModel):
    steps: List[str] = Field(
        description="""
        List of step by step re innovation plans.
        Each step contains detailed description about what change to made, what to add, what to remove, what to change, where to apply, how to apply etc,
        with proper description of goal aesthetic, look, also position, color description etc,
        so that any editor specialist can read it and apply these changes on appropriate position, item on the actual room image precisely.
        """,
        default=[]
    )
class ReviewResponse(BaseModel):
    approved: bool = Field(
        default=False,
        description="Whether the re innovation is approved or not."
    )
    comments: None| str = Field(
        default=None,
        description="Constructive criticism regarding the re innovation image quality, in case of rejection."
    )

class AgentState(BaseModel):
    max_critique_iterations:int=Field(
        default=3,
        description="Maximum number of critique iterations allowed."
    )
    current_critique_iteration:int=Field(
        default=0,
        description="Current critique iteration count."
    )
    generation_iteration:int=Field(
        default=0,
        description="Current generation iteration count."
    )
    planner_memory: List[SystemMessage | HumanMessage | AIMessage]=Field(
        default=[],
        description="History of previous edit patches"
    )
    current_edit_plan: Plans= Field(
        default=Plans(),
        description="Step by step plan of editing the actual image for re innovation"
    )
    user_edit_plan: str= Field(
        default="",
        description="Edit plan after user reviewed and edited the plan proposed by the agent."
    )
    current_gen_path: Path= Field(
        default=Path(),
        description="Path to the image generated"
    )
    critique_review: ReviewResponse= Field(
        default=ReviewResponse(),
        description="Critique review of the generated image."
    )
    #================================
    #Mendatory inputs
    room_image_path: Path= Field(
        description="Path to the image of the room to be re innovated."
    )
    theme_samples: List[Path] = Field(
        description="List of paths to sample images representing the desired theme for the room."
    )
    reinnovation_description: str = Field(
        description="A detailed description of the desired re-innovation for the room."
    )



def setupPlanPrompt(state: AgentState) -> AgentState:
    print("setting up planning prompt")
    state.planner_memory.extend(
        [
            SystemMessage(
                content="""
                You are an expert in room re innovation.
                You look into the room image provided provided, analyze its look, aesthetics, then you also look into the style inspiration images (if any), descriptions.
                Based on the desired look and aesthetic you make step by step plan of what changes to made.
                your step by step plan should not include modification of solid structures, 
                like for example: room size, window position, size, fixed light-fan-switch board positions, wiring, tiles etc.
                Your step by step re innovation plan should only contain modification, addition or removal of movable items. 
                some example are: portrait, table light, small table, bed, bedsheet, carpet, aquarium, twinkle lights etc etc.
                Give the output as json format.
                output example:
                ```
                { 
                    steps: [
                        "change the curtain to a beautiful red colored silky curtain with lily flower on it",
                        "apply maroon plastic paint on the light pink wall on the left side",
                    ]
                }
                ```
                """
            ),
            HumanMessage(
                content=[
                    dict(
                        create_text_block(text=f"""
                        The following is the image of the room, on which you need to plan re innovation:
                        """)
                    ),
                    dict(
                        local_image_message_content(path=state.room_image_path)
                    ),
                    dict(
                        create_text_block(text=f"""
                            The following is the description about the re innovation I want:
                            ---
                            {state.reinnovation_description}
                            ---

                            The following are some sample images of the theme I want:
                        """)
                    ),
                    *[
                        dict(
                            local_image_message_content(path=img_path)
                        )
                        for img_path in state.theme_samples
                    ]
                ]
            ),
        ]
    )
    return state



def planReinnovation(state: AgentState) -> AgentState:
    print("planning re innovation")
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.8, google_api_key=os.getenv("GOOGLE_FREE_API_KEY")).with_structured_output(Plans)
    state.current_edit_plan = cast(Plans,llm.invoke(state.planner_memory))
    print(f"Proposed step by step plan:\n{state.current_edit_plan.steps}")
    return state

def getUserReviewedPlan(state: AgentState) -> AgentState:
    print("getting user reviewed plan")
    # TODO:  This function can be implemented to get user-reviewed plans.
    state.user_edit_plan=("\n-- ").join(state.current_edit_plan.steps)
    print(f"Actionable plan proposed to user:\n{state.user_edit_plan}")
    state.planner_memory.append(
        HumanMessage(
            content=f"""
            The following is the step by step re innovation plan approved by the user:
            ---
            {state.user_edit_plan}
            ---
            """
        )
    )
    return state

def renderPlan(state: AgentState)->AgentState:
    print("rendering plan to generate re innovated image")
    message=[
        SystemMessage(
            content="""
                You are an image editor, expert in re innovating interior room design by following step by step editing plans given.
                you generate high fidality beautiful aesthetic room based on the image of the room, planning given to you.
                You only change the movable items in the scene. You do not add or modify extra item that is not present in the plan.
                You do not change rigit structure of the room and do not change the visual perspective of the given image.
            """
        ),
        HumanMessage(
            content=[
                dict(
                    create_text_block(
                        text="The following is the image, on which you need to apply the style changes and do the re innovation on:"
                    )
                ),
                dict(local_image_message_content(path=state.room_image_path)),
                dict(
                    create_text_block(
                        text=f"""
                        The following is the step by step plan of re innovation, apply these to the given image:
                        {state.user_edit_plan}
                        """
                    )
                )
            ]
        )
    ]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-image", temperature=0.9,google_api_key=os.getenv("GOOGLE_PAID_API_KEY"))
    response = llm.invoke(message)
    print("Received response from image generation LLM")
    image_folder=Path("output")
    file_name=f"generation_{state.generation_iteration}.png"
    save_path=image_folder/file_name
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            img_data = block["image_url"]["url"].split(",")[1]
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes))
            image.save(save_path)
            print(f"Image saved successfully as {file_name}")

    state.generation_iteration+=1
    state.current_gen_path=save_path
    state.planner_memory.append(
        AIMessage(
            content=[
                dict(
                    create_text_block(
                        text="The following is the re innovated image after applying the plan:"
                    )
                ),
                dict(
                    local_image_message_content(path=state.current_gen_path)
                )
            ]
        )
    )
    return state



def reviewCritique(state: AgentState) -> AgentState:
    print("reviewing and critiquing the generated image")
    messages=[
        SystemMessage(
            content="""
            You are an expert in judging the aesthetic quality of room re innovation.
            You look into the original room image, the re innovated image, and the plan that was followed to do the re innovation.
            Accept if the re innovation has been done according to the plan, if the re innovation looks decent enough, without any clear vizual artifact.
            Reject only if the re innovation does not follow the plan, does something out of plan, skip some plan, 
            or if the re innovation looks bad, unaesthetic, ugly, 
            or if the image of the re innovation has any vizual artifacts, distortions, visual glitches etc that make it sloppy, unrealistic,
            or if the re innovation changed any solid structure of the room etc.
            In case of rejection, give constructive criticism regarding what went wrong, what could have been better etc so that planning expert reads it and make better plan.
            You need to give feedback in json format. For example:
            ```
            #In case of acceptance
            {
                "approved": true,
                "comments": "The re innovation looks decent." #This is optional in case of acceptance
            }

            #In case of rejection
            {
                "approved": false,
                "comments": "The re innovation did extra changes not mentioned in the plan. The carpet looks distorted. Added separate window that is not actually present in the room."
            }
            ```
            """
        ),
        HumanMessage(
            content=[
                dict(
                    create_text_block(
                        text=f"""
                        The following was the plan of re innovation approved by the user:
                        ---------------
                        {state.user_edit_plan}
                        ---------------

                        The following is the original image of the room before re innovation:

                        """
                    )
                ),
                dict(
                    local_image_message_content(path=state.room_image_path)
                ),
                dict(
                    create_text_block(
                        text="""
                        The following is the re innovated image of the room after applying the re innovation plan:

                        """
                    )
                ),
                dict(
                    local_image_message_content(path=state.current_gen_path)
                )
            ]
        )
    ]
    llm= ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2,google_api_key=os.getenv("GOOGLE_FREE_API_KEY")).with_structured_output(ReviewResponse)
    state.critique_review=cast(ReviewResponse,llm.invoke(messages))
    print(f"Critique review: \nApproved={state.critique_review.approved}, \nComments={state.critique_review.comments}")
    if not state.critique_review.approved:
        state.planner_memory.append(
            AIMessage(
                content=f"""
                The critique rejected the re innovation. It gave the following feedback:
                ---------------
                {state.critique_review.comments}
                ---------------
                Now your job is to make a better step by step re innovation plan addressing these issues.
                Do not modify anything else which are not mentioned in the critique feedback.
                """
            )
        )
    return state


def editRouter(state: AgentState) -> Literal["ReviewCritique", "__end__"]:
    if state.current_critique_iteration < state.max_critique_iterations:
        state.current_critique_iteration+=1
        return "ReviewCritique"
    else:
        return "__end__"


def critiqueRouter(state: AgentState) -> Literal["ReinnovationPlanner", "__end__"]:
    if state.critique_review.approved:
        return "__end__"
    else:
        state.current_critique_iteration+=1
        return "ReinnovationPlanner"

workflow = StateGraph(AgentState)

workflow.add_node("SetupPlannerPrompt", setupPlanPrompt)
workflow.add_node("ReinnovationPlanner", planReinnovation)
workflow.add_node("GetUserReviewedPlan", getUserReviewedPlan)
workflow.add_node("ReinnovationVisualizer", renderPlan)
workflow.add_node("ReviewCritique", reviewCritique)

workflow.add_edge(START, "SetupPlannerPrompt")
workflow.add_edge("SetupPlannerPrompt", "ReinnovationPlanner")
workflow.add_edge("ReinnovationPlanner", "GetUserReviewedPlan")
workflow.add_edge("GetUserReviewedPlan", "ReinnovationVisualizer")
workflow.add_conditional_edges("ReinnovationVisualizer",editRouter)
workflow.add_conditional_edges("ReviewCritique",critiqueRouter)

app=workflow.compile()
if __name__ == "__main__":
    # # save the workflow graph image
    # png_data = app.get_graph().draw_mermaid_png()

    # with open("graph_workflow.png", "wb") as f:
    #     f.write(png_data)
    initial_state=AgentState(
        room_image_path=Path("samples/room.jpeg"),
        theme_samples=[
            Path("samples/gaming_theme.png"),
            Path("samples/modern_contemporary_g.jpeg"),
        ],
        reinnovation_description="""
        balances structure with warmth. The modern foundation provides clean lines, controlled symmetry, and a restrained color palette, ensuring the room feels intentional rather than chaotic. 
        Onto this base, bohemian elements introduce texture and emotion—layered fabrics, natural materials, plants, and artisanal décor soften the rigidity of modern design. 
        Lighting plays a critical role: modern, indirect or pendant lighting establishes clarity, while warm, decorative lights add intimacy. 
        The overall vibe is relaxed but confident—polished without feeling sterile, expressive without becoming messy.
        This blend suits someone who wants sophistication with personality, order with a visible human touch.
        """
    )
    final_state=app.invoke(initial_state)
    final_state_object=AgentState(**final_state)
    print("Final generated image path:", final_state_object.current_gen_path)