document.querySelector("#popupBtn").addEventListener("click", function () {
  document.querySelector(".popup").classList.add("active");
  document.querySelector(".head").classList.add("active");
  document.querySelector(".firstSection").classList.add("active");
  document.querySelector("main").classList.add("active");
  document.body.classList.add("active");
});


document.querySelector(".popup .close-btn").addEventListener("click", function () {
  document.querySelector(".popup").classList.remove("active");
  document.querySelector(".head").classList.remove("active");
  document.querySelector("main").classList.remove("active");
  document.querySelector(".firstSection").classList.remove("active");
  document.body.classList.remove("active");
});


function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].classList.remove("active");
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
      tablinks[i].classList.remove("active");
  }
  document.getElementById(tabName).classList.add("active");
  evt.currentTarget.classList.add("active");
}