// Event Listeners Function
document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("analyze-btn").addEventListener("click", analyzeEmail);
    document.getElementById("highlight-btn").addEventListener("click", highlightText);
    document.getElementById("rewrite-btn").addEventListener("click", rewriteText);
    document.getElementById("reanalysis-btn").addEventListener("click", reanalyzeEmail);
});

// Analyze Function
function analyzeEmail() {
    var email = document.getElementById("email").value;
    var analyzeBtn = document.getElementById("analyze-btn");
    // Add the loading animation class
    analyzeBtn.value = 'Loading...';
    analyzeBtn.disabled = true;
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/predict", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            var resultBoxSection = document.getElementById("result-box-section");
            var predictionText = response.prob > 0.7 ? '<span style="color: red;">Spam</span>' : '<span style="color: green;">Not Spam</span>';
            var spamProbability = response.percentage;
            var wordCount = response.result;
            var emoData = response.emo_data;
            // Update the HTML content
            resultBoxSection.innerHTML =
                "<h5>Prediction: " + predictionText + "</h5>" +
                "<h5>Spam Probability: " + spamProbability + " %</h5>" +
                "<div class='progress-bar'>" + "<div class='progress' id='progress-bar'></div>" + "</div>" +
                "<br>" +
                "<h5>Word Count: " + wordCount + "</h5>" +
                "<br>" +
                "<h5>Emotional Analysis:</h5>" +
                "<canvas id='emoChart'></canvas>";
            // Remove the loading animation class
            analyzeBtn.value = 'Analyze';
            analyzeBtn.disabled = false;
            // Update the progress bar width smoothly
            var progressBar = document.getElementById("progress-bar");
            progressBar.style.width = spamProbability + "%";
            progressBar.style.transition = "width 2s ease-in-out";
            // Tooltip on hover
            progressBar.addEventListener("mouseover", function () {
                progressBar.title = spamProbability + "%";
            });
            // Create the chart
            createEmoChart(emoData);
        }
    };
    xhr.send(JSON.stringify({ email: email }));
}

// Piechart Function
function createEmoChart(emoData) {
    var labels = emoData.map(function (data) {
        return data[0];
    });
    var scores = emoData.map(function (data) {
        return data[1];
    });
    var colors = ['#F06292', '#BA68C8', '#FFD54F', '#4DB6AC', '#FF8A65', '#7986CB', '#81C784', '#FFB74D'];

    if (emoData.length === 0) {
        // Display a default message or a blank chart
        var ctx = document.getElementById("emoChart").getContext("2d");
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.font = "16px Arial";
        ctx.textAlign = "center";
        ctx.fillText("No data available", ctx.canvas.width / 2, ctx.canvas.height / 2);
    } else {
        // Display the chart with data
        var ctx = document.getElementById("emoChart").getContext("2d");
        var chart = new Chart(ctx, {
            type: "pie",
            data: {
                labels: labels,
                datasets: [{
                    label: "Score",
                    data: scores,
                    backgroundColor: colors.slice(0, labels.length)
                }]
            },
            options: {
                responsive: true
            }
        });
    }
}

// Highlight Function
function highlightText() {
    var email = document.getElementById("email").value;
    var highlightBtn = document.getElementById("highlight-btn");
    // Add the loading animation class
    highlightBtn.value = 'Loading...';
    highlightBtn.disabled = true;
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/highlight_text", true);
    xhr.setRequestHeader("Content-Type", "application/json");

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var highlightedText = document.getElementById("highlighted-text");
            highlightedText.innerHTML = xhr.responseText;
            const questions = document.querySelectorAll(".question");

            var divElement = document.getElementById("rewrite");
            divElement.classList.remove("active");
        
            var divElement = document.getElementById("highlight-question-answer__accordion");
            divElement.classList.add("active");

            // Remove the loading animation class
            highlightBtn.value = 'Highlight';
            highlightBtn.disabled = false;
        }
    };
    xhr.send(JSON.stringify({ email: email }));
}

// Rewrite Function
function rewriteText() {
    var email = document.getElementById("email").value;
    var rewriteBtn = document.getElementById("rewrite-btn");
    var reanalysisBtn = document.getElementById("reanalysis-btn"); // New line

    rewriteBtn.value = 'Loading...';
    rewriteBtn.disabled = true;
    reanalysisBtn.disabled = true; // New line
    
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/rewrite_text", true);
    xhr.setRequestHeader("Content-Type", "application/json");

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var rewrittenText = document.getElementById("rewritten-text");
            rewrittenText.innerHTML = xhr.responseText;
            rewriteBtn.value = 'Rewrite';
            
            var divElement = document.getElementById("highlight-question-answer__accordion");
            divElement.classList.remove("active");
        
            var divElement = document.getElementById("rewrite");
            divElement.classList.add("active");

            rewriteBtn.disabled = false;
            reanalysisBtn.disabled = false; // New line
            analyzeEmail(); // Trigger the Analyze functionality
        }
    };

    xhr.send(JSON.stringify({ email: email }));
}

// Reanalysis Function
function reanalyzeEmail() {
    var rewrittenText = document.getElementById("rewritten-text").innerText;
    document.getElementById("email").value = rewrittenText;
    var rewrittenText = document.getElementById("rewritten-text");
    rewrittenText.innerHTML = '';
    var highlightedText = document.getElementById("highlighted-text");
    highlightedText.innerHTML = '';
    var divElement = document.getElementById("rewrite");
    divElement.classList.remove("active");
    var divElement = document.getElementById("highlight-question-answer__accordion");
    divElement.classList.remove("active");
    analyzeEmail(); // Trigger the Analyze functionality
}