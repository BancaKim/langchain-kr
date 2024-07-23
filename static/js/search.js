function searchCorp() {
    const searchType = document.getElementById('search_type').value;
    const searchValue = document.getElementById('search_value').value;
    
    fetch(`/baro_info?search_type=${searchType}&search_value=${searchValue}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('result').innerHTML = `<p>${data.error}</p>`;
            } else {
                document.getElementById('result').innerHTML = `
                    <p>Corp Code: ${data.corp_code}</p>
                    <p>Corp Name: ${data.corp_name}</p>
                    <p>Stock Code: ${data.stock_code}</p>
                    <p>Jurir No: ${data.jurir_no}</p>
                    <p>Bizr No: ${data.bizr_no}</p>
                    <p>Address: ${data.adres}</p>
                    <p>Home URL: ${data.hm_url}</p>
                    <p>IR URL: ${data.ir_url}</p>
                    <p>Phone No: ${data.phn_no}</p>
                    <p>Fax No: ${data.fax_no}</p>
                    <p>Induty Code: ${data.induty_code}</p>
                    <p>Establishment Date: ${data.est_dt}</p>
                    <p>Account Month: ${data.acc_mt}</p>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerHTML = `<p>Something went wrong</p>`;
        });
}


document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('search_value');
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