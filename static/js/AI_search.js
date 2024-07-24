function searchCorp() {
    const searchType = document.getElementById('search_type').value;
    const searchValue = document.getElementById('company_name').value;

    fetch(`/credit_companyinfo?search_type=${searchType}&search_value=${searchValue}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('result').innerHTML = `<p>${data.error}</p>`;
            } else {
                document.getElementById('result').innerHTML = `
                    <p>report_num: ${data.report_num}</p>
                    <p>corp_code: ${data.corp_code}</p>
                    <p>corp_name: ${data.corp_name}</p>
                    <p>report_nm: ${data.report_nm}</p>
                    <p>rcept_no: ${data.rcept_no}</p>
                    <p>rcept_dt: ${data.rcept_dt}</p>
                    <p>report_content: ${data.report_content}</p>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerHTML = `<p>Something went wrong</p>`;
        });
}


document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('company_name');
    const searchType = document.getElementById('search_type');
    const resultContainer = document.getElementById('result');

    searchInput.addEventListener('input', function () {
        const query = searchInput.value;
        const type = searchType.value;

        if (query.length >= 1) {
            fetch(`/autocomplete?search_type=${type}&query=${query}`)
                .then(response => response.json())
                .then(data => {
                    displaySuggestions(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        } else {
            resultContainer.innerHTML = ''; // 입력값이 없으면 결과를 비웁니다.
        }
    });

    function displaySuggestions(suggestions) {
        resultContainer.innerHTML = '';

        if (suggestions.length > 0) {
            const list = document.createElement('ul');
            list.classList.add('suggestions-list');

            suggestions.forEach(item => {
                const listItem = document.createElement('li');
                listItem.textContent = item;
                listItem.addEventListener('click', function () {
                    searchInput.value = item;
                    resultContainer.innerHTML = ''; // 선택하면 결과를 비웁니다.
                });
                list.appendChild(listItem);
            });

            resultContainer.appendChild(list);
        }
    }
});