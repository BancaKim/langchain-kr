// C:\Users\BIT\Desktop\파이썬\MVC\static\js\logout.js
function logout() {
    console.log('로그아웃 버튼 클릭됨');
    fetch('/logout', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('로그아웃 응답:', data);
        alert(data.message); // 로그아웃 메시지 표시
        window.location.href = '/'; // 홈 화면으로 리다이렉트
    })
    .catch((error) => {
        console.error('Error:', error);
        alert('로그아웃 중 오류가 발생했습니다.');
    });
}
