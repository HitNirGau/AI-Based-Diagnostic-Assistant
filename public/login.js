document.addEventListener("DOMContentLoaded", () => {
    if (localStorage.getItem('rememberMe')) {
        document.getElementById('username').value = localStorage.getItem('username') || '';
        document.getElementById('rememberMe').checked = true;
    }

    const loginForm = document.getElementById('loginForm');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const rememberMe = document.getElementById('rememberMe').checked;

        if (!username || !password) {
            alert("Username and password are required!");
            return;
        }

        try {
            const response = await fetch('http://localhost:5001/login', {  // 🔑 API route for login
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) {
                throw new Error('Invalid credentials or server error');
            }

            const data = await response.json();
            console.log('Response:', data);

            if (data.message === '✅ User authenticated!') {
                alert('Login Successful!');

                if (rememberMe) {
                    localStorage.setItem('username', username);
                    localStorage.setItem('rememberMe', true);
                } else {
                    localStorage.removeItem('username');
                    localStorage.removeItem('rememberMe');
                }

                window.location.href = '/dashboard';
            } else {
                alert('❌ Invalid credentials!');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('⚠️ Server error. Please try again later.');
        }
    });
});
