// -----------------------------------------------------------------------------
// INSTRUCTIONS

// go into full screen 
var full_screen = { 
	type: jsPsychFullscreen,
	fullscreen_mode: true
};

var initial_instructions_partone = {
	type: jsPsychInstructions,
	pages: ['<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
		'<h2> Thank you for agreeing to take part in this study! </h2><br>' +
		'We expect this study to take around <b>45 minutes</b> to complete. ' +
		'Since the hourly payment rate for this study is &pound7.50, ' +
		'you will earn <b>&pound6</b> if you complete the study. ' +
		'<br><br>IMPORTANT: <b>If you close the study tab or window in your browser your ' +
		'progress will be lost, and you will not be able to start the study again.</b> Please ' +
		'make sure you click the final <b>"Complete Study"</b> button at the end of the study, to ' +
		'submit your data back to Prolific and receive your payment.' +
		'<br><br>If you experience any technical difficulties, or have any other questions about ' +
		'the study, please get in touch with us at ' +
		'<a href=\"mailto:ion.mpc.cog@ucl.ac.uk\">ion.mpc.cog@ucl.ac.uk</a>, and we will aim ' +
		'to respond to you as quickly as possible. ' +
		'<br>' +
		'</div></div>',
		'<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
		'<h2> What is the study about? </h2><br>' +
		'In this experiment, you are going to see a number of <b> short emotional video clips </b> (5-10s). ' +
		'<br><br><b>These video clips contain material which can arouse strong emotions, both positive and negative. ' +
		'For instance, some clips may make you anxious, others happy or calm, and yet others disgusted or upset.</b> ' +
		'<br><br><b>We are interested in the emotions these video clips elicit in you.</b> ' +
		'Therefore, after each video clip we are going to ask you <b> how you are feeling in that moment</b>. ' +
		'<br>' +
		'</div></div>',
		'<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
		'<h2> What do I need to do? </h2>' +
		'You are going to watch different brief video clips. <br><br>After each video you are asked to indicate how much you feel certain emotions on a slider from "not at all" to "very". ' +
		'The sliders will be at the location you indicated the last time. <b>Each slider for each emotion has to be moved/clicked before you can continue.</b> ' +
		'If your emotions have not changed, you can click on the slider instead of moving it to another location. ' + 
		'<b>You have 30 seconds</b> to indicate how you feel. If you do not manage to respond in that time the experiment will move on. ' + 
		'<br><br><b>Please answer as truthfully as possible. It is critical for the scientific questions we are trying to answer.</b> ' +
		'<br><br><br>Next, we will ask you to indicate how you feel right now.' +
		'<br>' +
		'</div></div>'
	],
	show_clickable_nav: true,
	button_label_previous: 'go back',
	button_label_next: 'continue',
	on_start: function () {
		document.querySelector('body').style.backgroundColor = '#c8c8c8';
		saveStartData()
	},
};

var initial_instructions_parttwo = {
	type: jsPsychInstructions,
	pages: ['<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
		'<h2> Attention checks: </h2>' +
		'Thank you for telling us how you are feeling! ' +
		'<br><br>In addition to watching video clips and answering how you feel, ' +
		'we will sometimes apply <b>attention checks</b> which we explain to you on the next page. ' +
		'<b>If you answer incorrectly regularly, then we will assume that you have not paid attention and will not judge the submission as valid.</b> ' + 
		'<br><br>We are very grateful to our study participants for volunteering their time to help us ' +
		'with our research. Having those checks means that we can ' +
		'be more confident in the conclusions we can draw from online studies, and be more likely ' +
		'to be able to conduct these kind of studies in the future.' +
		'<br>' +
		'</div></div>'
	],
	show_clickable_nav: true,
	button_label_previous: 'go back',
	button_label_next: 'continue',
};

var initial_instructions_partthree = {
	type: jsPsychInstructions,
	pages: ['<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
		'<h2> How do the attention checks work? </h2><br>' +
		'<br>From time to time, you will be asked if you have seen a <b>black cross</b>. Sometimes, the black cross is briefly shown on the screen before the question and sometimes the black cross is not shown before the question. ' +
		'You simple have to answer if you have seen it or not. ' + 
		'<br><br><br><br>Next, we show you an example of such a situation. ' + 
		'<br>' +
		'</div></div>'
		],
	show_clickable_nav: true,
	button_label_previous: 'go back',
	button_label_next: 'continue',
	post_trial_gap: 1000,
};

var initial_instructions_partfour= {
	type: jsPsychInstructions,
	pages: ['Thank you! You got the attention check right.<br></br>',
		'<div class=\"row\"><div class=\"col-3\"></div><div class=\"col-6\">' +
		'Just to remind you one more time:' +
		'<br><br>The experiment lasts approximately <b>45 minutes</b>. On the <b>progress bar on the top</b> you can see how far you are.' +
		'<br><br><b>Please focus on the videos and do NOT close your eyes</b>. You can abort the experiment at any time by closing your browser but by that you stop partipating in the study.' +
		'<br><br> We also want to emphasize again, please <b>ONLY</b> continue if you are happy to view brief video clips with strong graphical content. ' +
		'</br></br><b>If you are ready to start the experiment press continue.</b>' +
		'</div></div>'
	],
	show_clickable_nav: true,
	button_label_previous: 'go back',
	button_label_next: 'continue',
	post_trial_gap: 1000,
};

var wrong_attention_checks = {
	type: jsPsychInstructions,
	allow_backward: false,
	show_clickable_nav: true,
	allow_keys: true,
	button_label_next: "continue",
	pages: [
		'<h2>Sorry, you did not get this right this time!</h2>' +
		'To check we have explained it clearly, please re-read the ' +
		'information and try again.' +
		'<br></br>'
	]
};

// ATTENTION CHECKS

var fix = {
	type: jsPsychHtmlKeyboardResponse,
	stimulus: '<h2> + </h2>',
	trial_duration: 1000,
	choices: "NO_KEYS",
};

var nothing = {
	type: jsPsychHtmlKeyboardResponse,
	stimulus: '',
	trial_duration: 500,
	choices: "NO_KEYS",
};

var attention_check_yes = true

var check = {
	type: jsPsychHtmlButtonResponse,
	stimulus: 'Did you just see a black cross?</br>',
	choices: ['yes', 'no'],
	trial_duration: 5000,
	on_finish: function (data) {
		if (data['response'] == 0) {
			attention_check_yes = true
		}
		else {
			attention_check_yes = false}
	}
};

// ATTENTION CHECK LOOP

var if_node = {
	timeline: [wrong_attention_checks],
	conditional_function: function (data) {
		if (attention_check_yes == true) {
			return false;
		} else {
			return true;
		}
	}
}

var loop_node = {
	timeline: [initial_instructions_partthree, fix, check, if_node],
	loop_function: function (data) {
		if (attention_check_yes == true) {
			return false;
		} else {
			return true;
		}
	}
};

var pilot_instructions = {
	type: jsPsychSurveyText,
	questions: [
		{
			prompt: 'As we are at the stage of piloting this study,</br>please let us know all your thoughts about the experiment you just did.',
			rows: 5,
		}
	]
}

// INSTRUCTIONS QUESTIONNAIRES

var questionnaire_instructions= {
	type: jsPsychInstructions, 
	pages: [
		'Thank you very much for completing the study!' +
		'</br></br> Before the end, we need to ask you a <b>set of questions about your general well-being!</b>' + 
		'</br></br><b>Please answer these as truthfully as possible.</br> It is critical for the scientific questions we are trying to answer.</b></br></br></br>'], 
	show_clickable_nav: true,
	button_label_previous: '',
	button_label_next: 'continue',
	post_trial_gap: 1000
};


// END

var end_screen = {
	type: jsPsychHtmlButtonResponse,
	choices: ['Complete Study'],
	is_html: true,
	stimulus: 'You have finished whole the experiment. <br>Thank you for your contribution to science. <br><br><b> PLEASE CLICK COMPLETE STUDY TO SUBMIT THE STUDY TO PROLIFIC </b>.<br></br>',
	on_start: function () {
		saveEndData();
		document.querySelector('body').style.backgroundColor = '#c8c8c8';
	},
	on_finish: function(){
		window.location = "https://app.prolific.co/submissions/complete?cc=COK3SSQT"; 
	},
};


