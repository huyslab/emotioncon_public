
	// Sign in
	// firebase.auth().signInAnonymously(); //
	firebase.auth().signInAnonymously().catch(function(error) {
		// Handle Errors here.;
		var errorCode = error.code;
		var errorMessage = error.message;
		console.log(errorCode);
		console.log(errorMessage);
	});

	// When signed in, get the user ID //
	var uid; // User ID 
	firebase.auth().onAuthStateChanged(function(user) {
	  if (user) {
		uid = user.uid;
		console.log('found...');
		console.log(uid);
	  } else {
		 uid = 'not found';
		console.log('not found');
	  }
	});

	//if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
	//	 alert("Sorry, this experiment does not work on mobile devices");
	//	 document.getElementById('consent').innerHTML = "";
	//}

