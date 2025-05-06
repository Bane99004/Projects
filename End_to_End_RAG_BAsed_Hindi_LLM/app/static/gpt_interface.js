document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("promptForm");
  form.addEventListener("submit", sendPrompt);
});

async function sendPrompt(event) {
  event.preventDefault();  // ⛔️ Stop form from reloading the page

  const promptInput = document.getElementById("promptInput");
  const prompt = promptInput.value.trim();
  const answerDiv = document.querySelector(".answer");
  const questionDiv = document.querySelector(".question");

  if (!prompt) {
    alert("Please enter a prompt!");
    return;
  }

  questionDiv.innerHTML = `<strong>You:</strong> ${prompt}`;
  answerDiv.innerHTML = `<em>GPT is thinking...</em>`;

  const requestData = { prompt: prompt, max_length: 70 };

  try {
    const response = await fetch("http://13.201.77.21:8000/gptinterface/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData)
    });

    const data = await response.json();

    if (response.ok) {
      answerDiv.innerHTML = `<strong>GPT:</strong> ${data.response}`;
    } else {
      answerDiv.innerHTML = "⚠️ Failed to generate response";
    }
  } catch (error) {
    answerDiv.innerHTML = "❌ Failed to connect to the server";
    console.error("Error:", error);
  }

  promptInput.value = "";
}
