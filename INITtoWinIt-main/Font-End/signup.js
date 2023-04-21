document.getElementById("toggle-password").addEventListener("click", function () {
    const passwordInput = document.getElementById("password");
    if (passwordInput.type === "password") {
        passwordInput.type = "text";
    } else {
        passwordInput.type = "password";
    }

});

// Beginning Of Typing JavaScript

const dynamicTxt = document.querySelector(".dynamic-txt");
const blinkingCursor = document.querySelector(".blinking-cursor");
const text = ["UniFace is cutting-edge technology that specializes in developing highly advanced facial recognition technology for secure access control at events and restricted locations. With UniFace's advanced AI technology, you can be sure that only authorized individuals gain entry to your event or location, thereby eliminating the risk of unauthorized access and ensuring the safety of your guests and staff. With UniFace, you can take your event security to the next level and provide your guests with a seamless and secure experience."];

let maxWidth = 0;
text.forEach(line => {
  const lineWidth = line.length * 10;
  if (lineWidth > maxWidth) {
    maxWidth = lineWidth;
  }
});
dynamicTxt.style.width = `${maxWidth}px`;

let wordIndex = 0;
let txtIndex = 0;
let isDeleting = false;
let delay = 40;

typeWords();

function typeWords() {
  const currentWordIndex = wordIndex % text.length;
  const currentWord = text[currentWordIndex];

  if (!isDeleting) {
    dynamicTxt.textContent = currentWord.substring(0, txtIndex);
    txtIndex++;

    if (txtIndex > currentWord.length) {
      isDeleting = true;
      delay = 1000;
    }
  } else {
    dynamicTxt.textContent = currentWord.substring(0, txtIndex);
    txtIndex--;

    if (txtIndex === 0) {
      isDeleting = false;
      wordIndex++;
      delay = 200;
    }
  }
  blinkingCursor.style.display = "none";

  setTimeout(typeWords, delay);
}