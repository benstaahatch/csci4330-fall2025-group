**This is fine**
This semester I've been diving deeper into machine learning through my Neural Networks course, specifically working with PyTorch and learning how different models behave during testing, training, and validation.

As l've started to understand these concepts more clearly, it made me reflect on a parallel computing project my team and I previously worked on last fall. In that project, we researched how high-performance computing using MPl and OpenMP could accelerate machine learning models written in C++.

-Feedback
**This blob of text is too much. Get to the point and just make a simple linked in post that points to the repository. You need to cut this way down. Originally the results were indeed inflated and what I learned is blah blah blah blah.**

We compared Logistic Regression, Random Forest, and Neural Networks to see which models benefited most from parallel processing. Logistic Regression worked well as a baseline, but it did not gain much from parallelization on our dataset. Neural Networks introduced synchronization and update overhead that made our CPU-based parallelism less effective. Random Forest ended up being the strongest candidate because each tree can be trained independently, making it naturally better suited for parallel computing.
Because of that, we selected Random Forest for the final hybrid implementation. The hybrid version used MPI to distribute trees across ranks, OpenMP to parallelize work inside each rank, and used histogram binning to speed up split finding.
The results that we initially reported were extremely high, close to 99% accuracy and F1 score. At first, that looked like a major success, but now I want to revisit the project with a more critical mindset. The dataset we used was relatively small: the Pima Indians Diabetes dataset which contains 768 samples and eight numerical features. Since it is a smaller classification dataset, I want to better understand whether those results reflect true model performance or whether they might have been inflated by overfitting, data leakage, or evaluating on the same data used for training.
I plan to rebuild the environment, pull the repository back onto my machine, rerun the experiments, and document what I find. My next goal is to move beyond this smaller binary classification dataset and eventually test the system on larger and more complex datasets where scalability and validation matter even more.
For me, this next phase is about learning how to validate high-performance machine learning systems properly, not just how to make them run faster. I'm curious to see whether the original results hold up under a more careful evaluation process.
I'm planning to document the process as I revisit the code, rerun the experiments and verify the results. Once I clean up the repo and confirm the evaluation pipeline, I'll share a follow-up with what I find.

