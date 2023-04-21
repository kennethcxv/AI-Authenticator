const passwordInput = document.getElementById("password");
const togglePassword = document.getElementById("toggle-password");

togglePassword.addEventListener("click", () => {
  if (passwordInput.type === "password") {
    passwordInput.type = "text";
    togglePassword.src = "Images/Hide Password Icon.png";
  } else {
    passwordInput.type = "password";
    togglePassword.src = "Images/Show Password Icon.png";
  }
});

// Beginning Of Typing JavaScript

const dynamicTxt = document.querySelector(".dynamic-txt");
const blinkingCursor = document.querySelector(".blinking-cursor");
const text = ["UniFace is cutting-edge technology that specializes in developing highly advanced facial recognition technology for secure access control at events and restricted locations. With UniFace's advanced AI technology, you can be sure that only authorized individuals gain entry to your event or location, thereby eliminating the risk of unauthorized access and ensuring the safety of your guests and staff."];

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
let delay = 40;

typeWords();

function typeWords() {
  const currentWordIndex = wordIndex % text.length;
  const currentWord = text[currentWordIndex];

  dynamicTxt.textContent = currentWord.substring(0, txtIndex);
  txtIndex++;

  if (txtIndex > currentWord.length) {
    delay = 1000;
  } else {
    delay = 40;
  }

  blinkingCursor.style.display = "none";
  setTimeout(typeWords, delay);
}