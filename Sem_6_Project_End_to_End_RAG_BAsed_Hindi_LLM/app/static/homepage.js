
document.addEventListener('DOMContentLoaded', () => {
  // Scroll animation
  const sections = document.querySelectorAll('section');
  const navItems = document.querySelectorAll('.nav-list ul li');
  
  // Highlight nav item on scroll
  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.clientHeight;
      if (pageYOffset >= (sectionTop - sectionHeight/3)) {
        current = section.getAttribute('id');
      }
    });
    
    navItems.forEach(item => {
      item.classList.remove('active');
      if(item.textContent.toLowerCase() === current) {
        item.classList.add('active');
      }
    });
  });
  
  // Swipe functionality
  const swipeHeading = document.getElementById('swipeHeading');
  const leftSwipe = document.querySelector('.leftSwipe');
  const rightSwipe = document.querySelector('.rightSwipe');
  const content = document.querySelector('.midSecondScroll');
  
  const section = [
    {
      heading: 'Our Mission',
      content: `
        <div class="Us">
          <p>We are a team of AI researchers, engineers, and data scientists dedicated to advancing artificial intelligence in a way that benefits enterprises and society. Our expertise spans machine learning, natural language processing, and data engineering.</p>
        </div>
        <div class="divider"></div>
        <div class="aims">
          <p>Our commitment is to develop sophisticated AI systems that integrate seamlessly with enterprise workflows, enhancing decision-making capabilities while maintaining the highest standards of data security and ethical considerations.</p>
        </div>
      `
    },
    {
      heading: 'Our Vision',
      content: `
        <div class="Us">
          <p>We envision a future where AI augments human capabilities across industries, enabling unprecedented levels of productivity and innovation while addressing complex global challenges.</p>
        </div>
        <div class="divider"></div>
        <div class="aims">
          <p>By 2026, we aim to be at the forefront of responsible AI development, setting industry standards for transparency, explainability, and ethical implementation of enterprise AI solutions.</p>
        </div>
      `
    },
    {
      heading: 'Our Values',
      content: `
        <div class="Us">
          <p>Excellence in research and engineering is the foundation of everything we do. We prioritize scientific rigor and technical precision in all our AI developments.</p>
        </div>
        <div class="divider"></div>
        <div class="aims">
          <p>We believe in transparent AI systems where decisions can be explained and understood. Our commitment to ethics guides our approach to data privacy, security, and algorithmic fairness.</p>
        </div>
      `
    }
  ];
  
  let currentIndex = 0;
  
  function updateContent() {
    swipeHeading.textContent = sections[currentIndex].heading;
    content.innerHTML = sections[currentIndex].content;
    content.style.animation = 'fadeSlide 0.5s ease forwards';
    setTimeout(() => {
      content.style.animation = '';
    }, 500);
  }
  
  leftSwipe.addEventListener('click', () => {
    currentIndex = (currentIndex - 1 + sections.length) % sections.length;
    updateContent();
  });
  
  rightSwipe.addEventListener('click', () => {
    currentIndex = (currentIndex + 1) % sections.length;
    updateContent();
  });
});
