
// -----------------------------------------------------------------------------
// FUNCTIONS TO SAVE DATA 
//


// Enable persistence
firebase.firestore().enablePersistence()
.catch(function (err) {
	if (err.code == 'failed-precondition') {
		// Multiple tabs open, persistence can only be enabled
		// in one tab at a a time.
	} else if (err.code == 'unimplemented') {
		// The current browser does not support all of the
		// features required to enable persistence
	};
});

var db = firebase.firestore();

// function to save consent
var saveConsent = function(){
	db.collection(collname).doc(uid).set({
		uid_firebase: uid, // firebase ID 
		subjectID_prolific: subjectID,  // prolific ID 
		studyID_prolific: studyID,  // prolific ID
		consent_obtained: 'yes',
		intervention_condition: random_intervention,
		videoset_condition: random_video_set,
		date: new Date().toLocaleDateString()
	}); 
};

// function to save initial data 
var saveStartData = function(){
	var starttime = jsPsych.getStartTime(); // record new date and start time
	db.collection(collname).doc(uid).update({
		start_time_jspsych: starttime, 
		completed: 0
	}); 
};

var saveSetup = function (timeline) {
	db.collection(collname).doc(uid).collection('setup').doc('timeline').set({
		timeline:JSON.stringify(timeline), 
	});
	db.collection(collname).doc(uid).collection('setup').doc('variables').set({
		collname	: 		collname,
		nTrials   	: 		nTrials,
		dofullscreen: 	dofullscreen,
	});
	// this is just to initialize the taskdata collection and the data and datadump documents
	db.collection(collname).doc(uid).collection('taskdata').doc('data').set({ foo: 1, });
	// this is just to initialize the questionnaire collection and the data and datadump documents
	db.collection(collname).doc(uid).collection('questionnaire').doc('data').set({ foo: 1, });
};

// function to save the task data
// save only every writefrequency trials 
// needs an index of trial number passed to it

var saveTaskData = function () {
	var length_quiz = jsPsych.data.get().filter({ trial_type: 'survey-multi-choice' }).select('response').values.length
	var datatosave = {
		slider_rt: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'html-multiple-slider-response' }).select('rt').values),
		slider_time: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'html-multiple-slider-response' }).select('time_elapsed').values),
		video_path: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'video-keyboard-response' }).select('stimulus').values),
		video_time: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'video-keyboard-response' }).select('time_elapsed').values),
		check_time: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'html-button-response' }).select('rt').values),
		check_response: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'html-button-response' }).select('response').values),
		intervention_sucess_time: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-multi-choice' }).select('rt').values[length_quiz-1]),
		intervention_sucess_response: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-multi-choice' }).select('response').values[length_quiz - 1]),
		intervention_improvement_response: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-text' }).select('response').values),
		intervention_improvement_time: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-text' }).select('rt').values),
		instruction_view_time: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'instructions' }).select('view_history').values)
	};
	stimuli = jsPsych.data.get().filter({ trial_type: 'html-multiple-slider-response' }).select('stimulus').values;
	responses = jsPsych.data.get().filter({ trial_type: 'html-multiple-slider-response' }).select('response').values;
	for (s = 0; s < stimuli.length; s++) {
		datatosave['stimulus' + (s + 1)] = JSON.stringify(stimuli[s]);
		datatosave['response' + (s + 1)] = JSON.stringify(responses[s]);
	};

	db.collection(collname).doc(uid).collection('taskdata').doc('data').set(datatosave);
	console.log(datatosave)
};

// function to save the task data
// save only every writefrequency trials  
// needs an index of trial number passed to it 
var saveQuestionnaireData = function () {
	var datatosave = {
		responses: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-likert' }).select('response').values), 
		trial_type: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-likert' }).select('trial_type').values),
		rt: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-likert' }).select('rt').values),
		question_order: JSON.stringify(jsPsych.data.get().filter({ trial_type: 'survey-likert' }).select('question_order').values),
	}
	db.collection(collname).doc(uid).collection('questionnaire').doc('data').update(datatosave);
};

var saveTaskDataDump = function(){
	var datatosave = { jsondatadump : jsPsych.data.get().json() }; 
	db.collection(collname).doc(uid).collection('taskdata').doc('datadump').set(datatosave)
};


// function to save final data 
var saveEndData = function(){
	saveTaskDataDump();
	saveTaskData();
	saveQuestionnaireData();
	db.collection(collname).doc(uid).update({
		total_time_jspsych: jsPsych.getTotalTime(),
		end_time_db: new Date().toLocaleTimeString(),
		completed: 1,
		terminatedEarly: 0
	});
};

// function to save incomplete data
var saveIncompleteData = function () {
	saveTaskDataDump();
	saveTaskData(true);
	db.collection(collname).doc(uid).update({
		total_time_jspsych: jsPsych.getTotalTime(),
		end_time_db: new Date().toLocaleTimeString(),
		completed: 0,
		terminatedEarly: 1
	})
};
