// RANDOMIZE INTO EMOTION REGULATION OR CONTROL INTERVENTION

var random_intervention = Number(Math.random() < 0.5)

// TEXT FOR INTERVENTION

var intervention_text = [
    [
        '<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
        '<h2>Emotion regulation strategy:</h2>' +
        '</br></br>' +
        'For the second part of the study, we would like you to <b>try out an emotion regulation technique</b> called distancing.' +
        '</br></br>' +
        'This technique involves viewing your emotions and thoughts as events passing in your mind rather than getting sucked in by them. ' +
        'We are interested in hearing whether and how well this works for you. It works for some people, but not for all.' +
        '</br></br>' +
        '<b>Please read the instructions carefully.</b> ' +
        'To ensure that you do not skip forwards accidentally, you can only click continue after a certain amount of time.' +
        '</br></br></br> ' +
        '</div ></div>',
        '<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
        '<h2>What do I have to do?</h2>' +
        '</br></br>' +
        'Usually, when an event evokes an emotion, we get sucked in.' +
        '</br></br>' +
        'One way of regulating emotions is to avoid getting sucked in, and instead <b>attempt to stand back ' +
        'and observe the emotion that happens to you as if it was a passing event.</b>' +
        '</br></br></br>' +
        'To illustrate this, we will walk you through a short mindfulness exercise ' +
        'called <b>"Leaves on a Stream"</b>.' +
        '</br></br></br>' +
        '</div ></div>',
        '</br></br>' +
        '<div class="center-content"><img src="prep/assets/imgs/eg1.png" style="width: 450px;"></img></div>' +
        '</br>' +
        'Imagine you are resting by the side of a ' +
        '</br>' +
        '<b>gently flowing stream watching the water flow.</b>' +
        '</br></br>',
        '</br></br>' +
        '<div class="center-content"><img src="prep/assets/imgs/eg1.png" style="width: 450px;"></img></div>' +
        '</br>' +
        '<b>Focus on the stream</b>, the sound of the water and other ambiance, ' +
        '</br>' +
        'the physical sensations, and anything else that comes to mind.' +
        '</br></br>',
        '<div class="center-content"><img src="prep/assets/imgs/eg2.png" style="width: 300px;"></img></div>' +
        '</br>' +
        'Imagine that there are <b>leaves</b> from trees, of all different shapes, sizes, and colors, ' +
        '</br>' +
        '<b>floating past on the stream</b> and you are just watching the leaves float on the stream.' +
        '</br></br>',
        '<div class="center-content"><img src="prep/assets/imgs/eg2.png" style="width: 300px;"></img></div>' +
        '</br>' +
        'The <b>stream does not stop</b>, it goes on continuously,</br>and the water can easily carry the leaves down/away.' +
        '</br></br>',
        '</br></br></br></br></br></br></br></br>' +
        '<h2>Now try to be aware of your emotions and thoughts.</h2>' +
        '</br></br></br>',
        '</br></br></br></br></br></br></br></br></br>' +
        'When an emotion or thought comes up, imagine you <b>place the thought on one of those leaves </b></br>' +
        'and that you are <b>watching the leave - carrying your emotion or thought - float away,</br>disappearing behind a corner or in the distance.</b>' +
        '</br></br></br>',
        '<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
        '</br></br></br></br></br>Some of the clips you are about to see are likely to elicit emotions. ' +
        'When the emotions start to come, try to <b>notice them without judgment.</b></br></br>' +
        'Emotions will intensify with each video clip. Try to feel them, allow them to come, ' +
        'and then also allow them to go again, like the leaves floating past.' +
        '</br></br></br>' +
        '</div ></div>',
        '</br></br></br></br></br></br></br></br></br>' +
        'Try to treat all your emotions the same, whether comfortable or uncomfortable.' +
        '</br>' +
        '<b>The goal is to become aware of your emotions — not to change or improve them.</b></br>' +
        '<b>Allow them to come, and then to go again.</b>' +
        '</br></br></br>',
        '<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
        '</br></br></br></br></br>Before you continue to the second part of watching video clips and reporting your emotions, we will ask you to <b>answer some quick questions</b>. ' +
        '</br></br>It is important for us that you have followed these instructions and the aim of the next phase of the experiment. ' +
        '<b>As such, if some answers are incorrect, we will ask you to read the instructions again.</b>' +
        '</br></br></br>' +
        '</div ></div>'
    ],
    [
        '<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
        '<h2>Relaxation phase:</h2></br></br>' +
        'Before you continue to the second part of the study, ' +
        'we would like to ask you to engage in a <b>relaxation exercise.</b>' +
        '</br></br> ' +
        '<b>Please read the next pages carefully.</b> To ensure that you do not skip ' +
        'forwards accidentally, you can only click continue after a certain amount of time.' +
        '</br></br></br> ' +
        '</div ></div>',
        '<h2>What do I have to do?</h2>' +
        '</br></br>' +
        'We are going to walk you through a relaxing exercise. </br>' +
        'Just read the next pages and try to relax.' +
        '</br></br></br>',
        '</br></br>' +
        '<div class="center-content"><img src="prep/assets/imgs/eg1.png" style="width: 450px;"></img></div>' +
        '</br>' +
        'Imagine you are resting by the side of a ' +
        '</br>' +
        '<b>gently flowing stream watching the water flow.</b>' +
        '</br></br>',
        '</br></br>' +
        '<div class="center-content"><img src="prep/assets/imgs/eg1.png" style="width: 450px;"></img></div>' +
        '</br>' +
        '<b>Focus on the stream</b>, the sound of the water and other ambiance, ' +
        '</br>' +
        'the physical sensations, and anything else that comes to mind.' +
        '</br></br>',
        '<div class="center-content"><img src="prep/assets/imgs/eg2.png" style="width: 300px;"></img></div>' +
        '</br>' +
        'Imagine that there are <b>leaves</b> from trees, of all different shapes, sizes, and colors, ' +
        '</br>' +
        '<b>floating past on the stream</b> and you are just watching the leaves float on the stream.' +
        '</br></br>',
        '<div class="center-content"><img src="prep/assets/imgs/eg2.png" style="width: 300px;"></img></div>' +
        '</br>' +
        'The <b>stream does not stop</b>, it goes on continuously,</br>and the water can easily carry the leaves down/away.' +
        '</br></br>',
        '</br></br></br></br></br></br></br></br>' +
        '<h2>Now keep thinking of the river and try to relax.</h2>' +
        '</br></br></br></br></br></br>',
        '</br></br></br></br></br></br></br></br></br>' +
        '<b>Imagine you are standing next to the river, and you are watching the leaves floating by,<br>' +
        'passing in front of you and then disappearing in the distance.</b>' +
        '</br></br></br></br></br></br></br>',
        '<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
        'For the next part of video clips, we would like to ask you to <b>keep doing ' +
        'what you have been doing in the first part: watching video clips and reporting your emotions.</b>' +
        '</br></br>Before you continue to the second part of watching video clips, ' +
        'we will ask you to <b>answer some quick questions</b>. ' +
        '</br></br>It is important for us that you have followed the relaxation exercise. ' +
        '<b>As such, if some answers are incorrect, we will ask you to read the instructions again.</b>' +
        '</br></br></br>' +
        '</div ></div>'
    ]
]


var quiz_text = [
    [
        '<b>1. The idea of the "Leaves on the Stream" is to...</b>',
        '<b>2. For the next part of video clips, I should...</b>'
    ],
    [
        '<b>1. The relaxation phase included...</b>',
        '<b>2. For the next part of video clips, I should...</b>'
    ]
]

var quiz_answsers = [
    [
        [
            "A) observe emotions getting stuck like leaves being trapped in a stream whirlpool.",
            "B) ignore emotions and focus on your breathing.",
            "C) observe emotions passing like leaves floating by on a stream."
        ],
        [
            "A) try to focus on my positive emotions and suppress negative emotions.",
            "B) try to observe my emotions and let them come and go.",
            "C) try to get sucked in by my emotions."
        ]
    ],
    [
        [
            "A) imagining laying on a wide meadow and looking into the sky.",
            "B) a breathing exercise.",
            "C) imagining standing next to a river and watching leaves floating by."
        ],
        [
            "A) try to supress my emotions.",
            "B) keep doing what I have been doing so far.",
            "C) try to remember the funniest video clip."
        ]
    ]
]

var continue_intervention_text = [
    [
        "<p><h2>Great you got the short quiz right!</h2></p>",
        '<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
        "<p><h2>What happens next?</h2></p></br>" +
        "Some people find this way of dealing with thoughts and emotions useful and some people do not find it useful. " +
        "What we are really interested to know is <b>if it works for you</b>.</br></br>" +
        "To see if it works for you, you can now give it a try with the next set of video clips. " +
        "At the end, we will ask for your view about this way of dealing with your thoughts and emotions.</br></br>" +
        "Please press continue to start with the second part of video clips.</br></br></br></div ></div>"
    ],
    [
        "<p><h2>Great you got the short quiz right!</h2></p>",
        "<p><h2>What happens next?</h2></p></br>" +
        "Some people find this relaxation phase useful, and some people do not find it useful.</br>" +
        "We are interested to know how it was for you. " +
        "After the next set of video clips,</br> we will ask for your view about this relaxation exercise.</br></br>" +
        "Please press continue to start with the second part of video clips.</br></br></br>"
    ]
]

var intervention_success_text = [
    'You finished the second part of the study! How successful, do you think, you were at complying with the instructions ' +
    '</br>for dealing with you emotions during the second part of the study?',
    'You finished the second part of the study! How successful, do you think, you were at relaxing during the relaxation phase?',
]


var intervention_questions_text = [
    'Please let us know how you would formulate an emotion regulation strategy such that it would better work for you.',
    'Please let us know what kind of relaxation techniques would work better for you.'
]


// intervention instructions //

var intervention_initial_instruction = {
    type: jsPsychInstructions,
    pages: ['<b>You have finished the first part of the study!</b>' +
        '<br><br>Please have a short break if you wish and press continue when you are ready to continue.<br><br>'
    ],
    show_clickable_nav: true,
    button_label_previous: 'go back',
    button_label_next: 'continue',
    post_trial_gap: 1000,
};

var intervention_instruction = {
    type: jsPsychInstructionsTime,
    pages: intervention_text[random_intervention],
    show_clickable_nav: true,
    button_label_previous: 'go back',
    button_label_next: 'continue',
    min_viewing_time: 5000,
};

var intervention_emotion_prompt = [
    '</br></br> <small><small><small><em> You observed your emotions and let them pass</br>like the leaves floating by on the stream.</br></br>' +
    '</small ></small ></em > How are you feeling right now?']

// QUIZ //

var quizQuestions = [
    {
        prompt: quiz_text[random_intervention][0],
        options: quiz_answsers[random_intervention][0],
        required: true,
        horizontal: false
    },
    {
        prompt: quiz_text[random_intervention][1],
        options: quiz_answsers[random_intervention][1],
        required: true,
        horizontal: false
    },
];

var sorryText = {
    type: jsPsychInstructions,
    allow_backward: false,
    show_clickable_nav: true,
    allow_keys: true,
    button_label_next: "continue",
    pages: ["<p><h2>Sorry, you didn’t get all the answers right this time!</h2></p>" +
        "<p>" +
        "Please re-read the information and try the quiz again." +
        "</p>"],
    post_trial_gap: 1000,
};

var nCorrect = 0;
var introQuiz = {
    type: jsPsychSurveyMultiChoice,
    questions: quizQuestions,
    data: {
        correct_answers: ["C)", "B)"]
    },
    randomize_question_order: false,
    button_label: "check answers",
    on_finish: function (data) {
        // compare answers to correct answers //
        nCorrect = 0;
        for (var i = 0; i < data.correct_answers.length; i++) {
            var questID = "Q" + i;
            if (data.response[questID].includes(data.correct_answers[i])) {
                nCorrect++;
            }
        }
        data.nCorrect = nCorrect;
    }
};

var intervention_node = {
    timeline: [sorryText],
    conditional_function: function (data) {
        if (nCorrect >= 2) {
            return false;
        } else {
            return true;
        }
    }
};

var intervention_loop = {
    timeline: [intervention_instruction, introQuiz, intervention_node],
    loop_function: function (data) {
        if (nCorrect >= 2) {
            return false;
        } else {
            return true;
        }
    }
};

var continue_intervention_instructions = {
    type: jsPsychInstructions,
    allow_backward: false,
    show_clickable_nav: true,
    allow_keys: true,
    button_label_next: "continue",
    pages: continue_intervention_text[random_intervention],
    post_trial_gap: 1000,
};


var intervention_success = {
    type: jsPsychSurveyMultiChoice,
    questions: [
        {
            prompt: intervention_success_text[random_intervention],
            options: ['Extremely successful.', 'Pretty successful.', 'I do not know.', 'Moderately successful.', 'Not at all successful.'],
            required: true
        }
    ]
}

var intervention_questions = {
    type: jsPsychSurveyText,
    questions: [
        {
            prompt: intervention_questions_text[random_intervention],
            rows: 5,
            required: true
        }
    ]
}
