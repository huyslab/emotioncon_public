## Experiment Details

This project involves an online task where participants are asked to rate their emotions while watching emotional videos. The task is run using the JsPsych library and data is stored on Firebase. Here are the details of the project:

- The videos used for the task can be downloaded from https://s3-us-west-1.amazonaws.com/emogifs/ in the `/public/videos/download_videos.sh` file.
- The videosequences were prespecified using the `categorization_videos.ipynb` notebook in the `../analyses` directory.
- The online task is located in the `/public` directory and can be accessed by opening the `index.html` file.
- The `index.html` file loads the necessary libraries and scripts, including JsPsych, Firebase, and the `emotionconexp.js` script which defines the task itself.
- The data is stored on Firebase using the `firestore.js` script which sets up Firestore for storing data. Note that proper permissions must be set up on Firestore in Firebase for this to work correctly.
- `index.html` runs `getconsentandrun.js` which gets participant consent and starts the task by running `runtask()`.
- To download the data, there is a Python script located in the `../analyses` directory.

Note that the `/public` folder needs to be hosted on Firebase. To do this, install Firebase from npm and deploy to the appropriate Firebase project.
