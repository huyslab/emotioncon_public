// DERS-18

var options = ["Almost Never<br>(0-10%)", "Sometimes<br>(11-35%)", "About Half the Time (35-65%)", "Most of the time (66-90%)", "Almost Always (91-100%)"];

var DERS = {
    type: jsPsychSurveyLikert,
    preamble: '<p max-width="500px"></br></br></br><b>Please choose the response that is most true for you.</p>',
    questions: [
        { prompt: "<b> I pay attention to how I feel.</b>",                                             name: "DERS_1", labels: options, required:true}, 
        { prompt: "<b>I have no idea how I am feeling.</b>",                                             name: "DERS_2", labels: options, required:true},
        { prompt: "<b> I have difficulty making sense out of my feelings.</b>",                         name: "DERS_3", labels: options, required:true},
        { prompt: "<b>I am attentive to my feelings.</b>",                                              name: "DERS_4", labels: options, required:true},
        { prompt: "<b> I am confused about how I feel.</b>",                                            name: "DERS_5", labels: options, required:true},
        { prompt: "<b>When I'm upset, I acknowledge my emotions.</b>",                                  name: "DERS_6", labels: options, required:true},
        { prompt: "<b>When I'm upset, I become embarrassed for feeling that way.</b>",                  name: "DERS_7", labels: options, required: true },
        { prompt: "<b>When I'm upset, I have difficulty getting work done.</b>",                        name: "DERS_8", labels: options, required: true },
        { prompt: "<b>When I'm upset, I become out of control.</b>",                                    name: "DERS_9", labels: options, required: true },
        { prompt: "<b>When I'm upset, I believe that I will remain that way for a long time.</b>",      name: "DERS_10", labels: options, required: true },
        { prompt: "<b>When I'm upset, I believe that I'll end up feeling very depressed.</b>",          name: "DERS_11", labels: options, required: true },
        { prompt: "<b>When I'm upset, I have difficulty focusing on other things.</b>",                 name: "DERS_12", labels: options, required: true },
        { prompt: "<b>When I'm upset, I feel ashamed with myself for feeling that way.</b>",            name: "DERS_13", labels: options, required: true },
        { prompt: "<b>When I'm upset, I feel guilty for feeling that way.</b>",                         name: "DERS_14", labels: options, required: true },
        { prompt: "<b>When I'm upset, I have difficulty concentrating.</b>",                            name: "DERS_15", labels: options, required: true },
        { prompt: "<b>When I'm upset, I have difficulty controlling my behaviors.</b>",                 name: "DERS_16", labels: options, required: true },
        { prompt: "<b>When I'm upset, I believe that wallowing in it is all I can do.</b>",             name: "DERS_17", labels: options, required: true },
        { prompt: "<b>When I'm upset, I lose control over my behaviors.</b>",                           name: "DERS_18", labels: options, required: true }
    ],
    on_finish: function () { saveQuestionnaireData() },
};

var timeline_DERS = [];
timeline_DERS.push(DERS);
