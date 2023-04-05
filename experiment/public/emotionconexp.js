// -----------------------------------------------------------------------------
// RUN EXPERIMENT

// define global variables 
var collname = 'emotionconexp' ; 	// under what name to save it 

var jsPsych = initJsPsych({
	show_progress_bar: true,
	message_progress_bar: 'progress',
	auto_update_progress_bar: true,
});

var debugging = false;
if (debugging == true) { var dofullscreen = false } else { var dofullscreen = true}
var fix_trials = design.fix_trials;
var check_trials = design.check_trials;

var nTrials = design.number_of_trials;
var random_video_set = Number(Math.random() < 0.5)

var timeline = [];  /* list of things to run */

if (dofullscreen == true) { timeline.push(full_screen); }

var baseline_emotions = {
	type: jsPsychHtmlMultipleSliderResponse,
	prompt: 'How are you feeling right now? </br></br>' +
		'<small><small><small><em> Please move the slider for each emotion </br> according to how you are feeling between </br>' +
		'you are not feeling this emotion at all (left) and ' +
		'</br> you are feeling this emotion very much (right) </br> and press continue. </small ></small ></small ></em > ',
	stimulus: design.emotion_array,
	require_movement: true,
	slider_width: 500,
	labels: ['not at all', 'very'],
};

timeline.push(initial_instructions_partone, baseline_emotions, initial_instructions_parttwo, loop_node, initial_instructions_partfour);

var last_location;

for (b = 0; b < 2; b++) {

	if (b == 1) {

		timeline.push(intervention_initial_instruction, intervention_loop, continue_intervention_instructions, baseline_emotions);
	};

	var video_code = design.video_sets[+(b == random_video_set)];

	for (t = 0; t < nTrials/2; t++) {

		var video_trial = {
			type: jsPsychVideoKeyboardResponse,
			stimulus: [
				'videos/' + video_code[t]
			],
			choices: "NO_KEYS",
			trial_ends_after_video: true,
			autoplay: true
		};

		var shuffledArray = jsPsych.randomization.repeat(design.emotion_array, 1);

		if (b == 1 & random_intervention == 0) {
			emotion_quiz_text = intervention_emotion_prompt
		}
		else { emotion_quiz_text = 'How are you feeling right now?' }

		var emotions = {
			type: jsPsychHtmlMultipleSliderResponse,
			prompt: emotion_quiz_text,
			slider_width: 500,
			stimulus: shuffledArray,
			t: t + b * (nTrials / 2 + 1),
			require_movement: true,
			trial_duration: 30000,
			labels: ['not at all', 'very'],
			on_start: function (emotions) {
				previous_stimuli = jsPsych.data.get().filter({ trial_type: 'html-multiple-slider-response' }).select('stimulus').values;
				next_stimuli = emotions.stimulus;
				responses = jsPsych.data.get().filter({ trial_type: 'html-multiple-slider-response' }).select('response').values;
				r = responses[emotions.t]
				x = 0
				while (r==null){
					x++;
					r = responses[emotions.t-x];
				}
				responses = r;
				previous_stimuli = previous_stimuli[emotions.t-x];
				var last_location = [];
				for (var s = 0; s < responses.length; s++) {
					idx = previous_stimuli.indexOf(next_stimuli[s]);
					last_location[s] = responses[idx];
				};
				emotions.slider_start = last_location;
			},
		};

		timeline.push(video_trial, emotions);

		if (fix_trials.includes(t + b * (nTrials / 2 + 1))) { timeline.push(fix, check) };
		if (check_trials.includes(t + b * (nTrials / 2 + 1))) { timeline.push(nothing, check) };

	};
}


timeline.push(intervention_success, intervention_questions, questionnaire_instructions)
timeline = timeline.concat(timeline_PHQ, timeline_GAD, timeline_DERS);
timeline.push(end_screen);

// now call jsPsych.init to run experiment

function runtask(uid) {
	saveSetup(timeline);
	jsPsych.run(timeline);
}

