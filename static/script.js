document.getElementById("audioForm").onsubmit = function(event) {
    event.preventDefault();
    let formData = new FormData();
    formData.append("audio", document.getElementById("audioInput").files[0]);

    fetch("/predict_audio", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => document.getElementById("audioResult").innerText = "Sentiment: " + data.sentiment);
};

document.getElementById("imageForm").onsubmit = function(event) {
    event.preventDefault();
    let formData = new FormData();
    formData.append("image", document.getElementById("imageInput").files[0]);

    fetch("/predict_image", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => document.getElementById("imageResult").innerText = "Emotion: " + data.emotion);
};

document.getElementById("textForm").onsubmit = function(event) {
    event.preventDefault();
    let text = document.getElementById("textInput").value;

    fetch("/predict_text", { method: "POST", body: JSON.stringify({text}), headers: { "Content-Type": "application/json" } })
    .then(response => response.json())
    .then(data => document.getElementById("textResult").innerText = "Result: " + data.result);
};
