burger = document.querySelector('.burger')
navbar = document.querySelector('.navbar')
rightnav = document.querySelector('#popupBtn')
navList = document.querySelector('.nav-list')
discript_one = document.querySelector('.discript-one')
discript_one = document.querySelector('.discript-two')

burger.addEventListener('click', ()=>{
  rightnav.classList.toggle('v-class-resp')
  navList.classList.toggle('v-class-resp')
  navbar.classList.toggle('h-nav-resp')
  
})
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry)=>{
    if (entry.isIntersecting){
      entry.target.classList.add('show');
    }
    else{
      entry.target.classList.remove('show');
    }
  });
});
const hiddenElements = document.querySelectorAll('.hidden');
hiddenElements.forEach((el) => observer.observe(el));