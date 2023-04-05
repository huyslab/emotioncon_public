// Consent form

document.getElementById('consent').innerHTML = " <div class=\"row\"> " +
    " 	<div class=\"col-3\"></div> " +
    " 	<div class=\"col-6\"><h2>Modular Tests of Cognitive Interventions</h2>" +
    " 		<p><b>Please read through the Participants Information Sheet carefully before starting the study and download or print it for your record." +
    "             After reading the Participants Information Sheet, please complete the consent form below before starting the study. If you need any further information to help " +
    "             you decide whether or not to take part, then please get in contact with the research team via" +
    "       <a href=\"mailto:ion.mpc.cog@ucl.ac.uk\">ion.mpc.cog@ucl.ac.uk\</a>.</b></p></div></div>" +
    " <div class=\"row\"> " +
    " 	<div class=\"col-3\"></div> " +
    "	<div class=\"col-6\"><iframe src=\"prep/PIS.pdf\" style=\"width: 100%; height: 900px\"></iframe></div></div>" +
    " <div class=\"row\"> " +
    " 	<div class=\"col-3\"></div> " +
    " 	<div class=\"col-6\"> " +
    " 		<p><b>To indicate your consent to take part in this study, please read the statements below and tick the box to" +
    "             tell us whether or not you agree with each statement. You can only take part in the study if you agree with all the statements.</b></p>  " +
    "  " +
    " 		</p>  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox1\">  " +
    " 		I have read and understood the Information Sheet (Version 2, 11/03/2022). I have had an opportunity to consider the information and what will be expected of me. I have also had the opportunity to ask questions which have been answered to my satisfaction. " +
    " 		<span class=\"checkmark\"></span>  " +
    " 		</label>  " +
    " 		<br> <br> " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox2\">  " +
    " 		I consent to the processing of my personal data for the purposes explained to me in the Information Sheet. I understand that my information will be handled in accordance with all applicable data protection legislation and ethical standards in research. " +
    " 		<span class=\"checkmark\"></span>  " +
    " 		</label> <br><br> " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox3\">  " +
    " 		I understand that I am free to withdraw from this study at any time without giving a reason and this will not affect my future medical care or legal rights. " +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox4\">  " +
    " 		I understand the potential benefits and risks of participating, the support available to me should I become distressed during the research, and whom to contact if I wish to lodge a complaint. " +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox5\">  " +
    " 		I understand the inclusion and exclusion criteria set out in the Information Sheet. I confirm that I meet the inclusion criteria.  " +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox6\">  " +
    " 		I understand that my anonymised personal data can be shared with others for future research, shared in public databases, and in scientific reports. " +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox7\">  " +
    " 		I understand that the data acquired is for research purposes and agree to it being kept and analysed even if and after I withdraw from the study. " +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox8\">  " +
    " 		I am aware of who I can contact should I have any questions or if any issues arise. " +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox9\">  " +
    " 		I understand that I will be shown videos with strong graphical content that are likely to elicit strong negative and positive emotions." +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<label class=\"container\"> " +
    " 		<input type=\"checkbox\" id=\"consent_checkbox10\">  " +
    " 		I voluntarily agree to take part in this study." +
    " 		<span class=\"checkmark\"></span> <br><br> " +
    " 		</label>  " +
    "  " +
    " 		<br><br>  " +
    " 		<button type=\"button\" id=\"start\" class=\"submit_button\">continue</button>  " +
    " 		<br><br> " +
    " 	</div> " +
    " 	<div class=\"col-3\"></div> " +
    " </div> ";

if (debugging == false) {

    var check_consent = function (elem) {
        if ($('#consent_checkbox1').is(':checked') && $('#consent_checkbox2').is(':checked') && $('#consent_checkbox3').is(':checked') && $('#consent_checkbox4').is(':checked') && $('#consent_checkbox5').is(':checked') && $('#consent_checkbox6').is(':checked') && $('#consent_checkbox7').is(':checked') && $('#consent_checkbox8').is(':checked') && $('#consent_checkbox9').is(':checked') && $('#consent_checkbox10').is(':checked')) {
            // When signed in, get the user ID
            firebase.auth().onAuthStateChanged(function (user) {
                if (user) {
                    uid = user.uid;
                    saveConsent(uid);
                    runtask(uid);
                }
            });
        } else {
            alert("Unfortunately you will not be able to participate in this research study if you do " +
                "not consent to the above. Thank you for your time.");
            return false;
        }
    };

    document.getElementById("start").onclick = check_consent;

} else {
    document.getElementById("start").onclick = function () {
        runtask(1);
    };
};


