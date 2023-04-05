//PHQ_9

  // definiting two different response scales that can be used.
  var options = ["Not at all", "Several days", "More than</br>half the days", "Nearly every day"];

  var PHQ = {
      type: jsPsychSurveyLikert,
      preamble: '<p max-width="500px"></br></br></br><b>Over the last 2 weeks</b>, how often have you been bothered by any of the following problems?</p>',
      questions: [
          {prompt: "<b>Little interest or pleasure in doing things</b>",         name: "PHQ9_1", labels: options, required:true}, 
          {prompt: "<b>Feeling down, depressed, or hopeless</b>",                name: "PHQ9_2", labels: options, required:true},
          {prompt: "<b>Trouble falling/staying asleep or sleeping too much</b>", name: "PHQ9_3", labels: options, required:true},
          {prompt: "<b>Feeling tired or having little energy</b>",               name: "PHQ9_4", labels: options, required:true},
          {prompt: "<b>Poor appetite or overeating</b>",                         name: "PHQ9_5", labels: options, required:true},
          {prompt: "<b>Feeling bad about yourself or that you are a failure or have let yourself or your family down</b>", name: "PHQ9_6", labels: options, required:true},
          {prompt: "<b>Trouble concentrating on things, such as reading the newspaper or watching television.</b>", name: "PHQ9_7", labels: options, required:true},
          {prompt: "<b>Moving or speaking so slowly that other peope could have noticed.</br> Or the opposite; being so fidgety or restless that you have been moving around a lot more than usual.</b>", name: "PHQ9_8", labels: options, required:true},
          {prompt: "<b>Thoughts that you would be better off dead or of hurting yourself in some way.</b>", name: "PHQ9_9", labels: options, required:true},
      ],
      on_finish: function () { saveQuestionnaireData() },
};

var difficulty = ["Not difficult at all", "Somewhat difficult", "Very diffucult", "Extremely difficult"];
  //difficulty
var diff = {
    type: jsPsychSurveyLikert,
    questions: [
        { prompt: "If you had any of the problems above, how difficult have these problems made it for you to do your work, take care of things at home, or get along with other people?", name: "PHQ9_diff", labels: difficulty, required: true }, 
    ],
    on_finish: function () { saveQuestionnaireData() },
};
  
 var timeline_PHQ = [];
 timeline_PHQ.push(PHQ);
 timeline_PHQ.push(diff);
