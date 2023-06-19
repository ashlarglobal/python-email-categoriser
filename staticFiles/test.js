function highlight (){
    let text_highlight = document.getElementById('text_highlight').value;
    let paragraph = document.getElementById('paragraph').value;
    console.log(text_highlight);
    text_highlight = text_highlight.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");
    let pattern = new RegExp(`${text_highlight}`, "gi");
    paragraph.innerHTML = paragraph.textContent.replace(pattern, match => `<mark>${match}</mark>`);
}