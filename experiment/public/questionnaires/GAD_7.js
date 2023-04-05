// GAD-7

var options = ["Not at all", "Several days", "More than</br>half the days", "Nearly every day"];

var GAD = {
    type: jsPsychSurveyLikert,
    preamble: '<p max-width="500px"></br></br></br><b>Over the last 2 weeks</b>, how often have you been bothered by any of the following problems?</p>',
    questions: [
        {prompt: "<b>Feeling nervous, anxious, or on edge</b>",                     name: "GAD_1", labels: options, required:true}, 
        {prompt: "<b>Not being able to stop or control worrying</b>",               name: "GAD_2", labels: options, required:true},
        {prompt: "<b>Worrying too much about different things</b>",                 name: "GAD_3", labels: options, required:true},
        {prompt: "<b>Trouble relaxing</b>",                                         name: "GAD_4", labels: options, required:true},
        {prompt: "<b>Being so restless that it is hard to sit still</b>",           name: "GAD_5", labels: options, required:true},
        {prompt: "<b>Becoming easily annoyed or irritable</b>",                     name: "GAD_6", labels: options, required:true},
        {prompt: "<b>Feeling afraid, as if something awful might happen.</b>",      name: "GAD_7", labels: options, required:true},

    ],
    on_finish: function () { saveQuestionnaireData() },
};

var timeline_GAD = [];
timeline_GAD.push(GAD);
