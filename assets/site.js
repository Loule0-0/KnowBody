const navToggle = document.querySelector('.nav-toggle');
const siteNav = document.querySelector('.site-nav');

navToggle?.addEventListener('click', () => {
  const isOpen = siteNav.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(isOpen));
});

siteNav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    siteNav.classList.remove('open');
    navToggle?.setAttribute('aria-expanded', 'false');
  });
});

const dialog = document.querySelector('#figure-dialog');
const dialogImage = dialog?.querySelector('img');

document.querySelectorAll('[data-zoom]').forEach((button) => {
  button.addEventListener('click', () => {
    if (!dialog || !dialogImage) return;
    dialogImage.src = button.dataset.zoom;
    dialogImage.alt = button.dataset.alt || '';
    dialog.showModal();
  });
});

dialog?.querySelector('.dialog-close')?.addEventListener('click', () => dialog.close());
dialog?.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close();
});

const copyButton = document.querySelector('#copy-citation');
copyButton?.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex')?.innerText || '';
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
  } catch {
    copyButton.textContent = 'Select text';
  }
});
