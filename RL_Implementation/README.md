<h1> Haiku Transformer </h1>
Transform free verse poems into haikus using reinforcement learning (RL). This project combines natural language generation with RL to edit free verse poetry iteratively until it conforms to the haiku form's constraints.

<h2>Goal</h2>

- Use reinforcement learning to transform free verse poems into haikus that follow the traditional 5-7-5 syllable structure while preserving the original poem's themes and imagery. <br>
- Why?
    - There are very few datasets of Image to Poems and there are none that are image to haiku
    - We want to show a way to transform any image into a poem with a fixed deterministic style without the need of a specific dataset for it

<h2>Project Overview</h2>
<h3> Define Constraints </h3>

- Syllable Count: Reward the model for aligning with the haiku's 5-7-5 syllable structure.<br>
- Content Relevance: Maintain the poem's original themes and imagery.<br>
- Line Structure: Adhere to the haiku’s three-line structure, without mid-line breaks.<br>

<h3>Generate Free Verse Poems</h3>

- Use the img2poem module to generate free verse poems as initial drafts.<br>

<h3> Setup Detector and Prompter Modules </h3>

- Detector: Identifies the line or word that needs revision to better fit the haiku structure. <br>
- Prompter: Suggests alternatives that are compatible with the haiku’s syllabic and thematic constraints. <br>

<h3>Setup Evaluation Modules </h3>

- Pre-trained Language Models: Ensure semantic and stylistic coherence by using embedding similarity and contextual constraints. <br>
- Syllable Counter: Actively guides the Prompter to reduce the difference between the current syllable count and the target haiku syllable count.<br>

<h3> Reward Function Modeling  </h3>

- Design a reward function that considers syllable count, structure adherence, and thematic relevance. <br>

<h3>Training </h3>

- Use Proximal Policy Optimization (PPO) or Natural Policy Gradient (NPG) for iterative revisions until the poem matches the haiku structure. <br>

<h2> Setup and Timeline </h2>
<h3> Week 1: Setup </h3>
- Create a dataset that pairs 200 free verses to 200 haikus using ChatGPT
- Fine-tune the language model on haikus, if needed. <br>
- Integrate the syllable counter for syllabic accuracy. <br>
- Develop the reward function and test initial criteria for semantic and structural constraints. <br>
- Implement the Detector and Prompter modules. <br>
<h3> Week 2: Optimization </h3> 

- Refine and tune the reward function for improved performance. <br>
- Debug and optimize the Detector and Prompter to ensure smooth interaction with the syllable counter and language model. <br>
<h3> Week 3: Training </h3>

- Train the RL agent on the haiku transformation task, using iterative revisions to achieve alignment with haiku constraints.<br>
<h3> Week 4: Finalization </h3>

- Conduct final evaluations and testing on diverse poems. <br>
- Integrate additional elements, including Img2Poem and Poetry2Speech, to complete the pipeline. <br>