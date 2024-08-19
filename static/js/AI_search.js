document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('company_name');
    const resultContainer = document.getElementById('autocomplete-results');
    const searchTypeSelect = document.getElementById('search_type');

    searchInput.addEventListener('input', function () {
        const query = searchInput.value;
        const searchType = searchTypeSelect.value;  // Get the selected search type

        if (query.length >= 1) {
            fetch(`../autocomplete?query=${query}&search_type=${searchType}`)
                .then(response => response.json())
                .then(data => {
                    displaySuggestions(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        } else {
            resultContainer.innerHTML = ''; // Clear results if input is empty
            resultContainer.classList.add('hidden'); // Hide the results
        }
    });

    function displaySuggestions(suggestions) {
        resultContainer.innerHTML = '';

        if (suggestions.length > 0) {
            const list = document.createElement('ul');
            list.classList.add('list-none', 'p-0', 'm-0');

            suggestions.forEach(item => {
                const listItem = document.createElement('li');
                listItem.textContent = item;
                listItem.classList.add('p-2', 'cursor-pointer', 'hover:bg-gray-100');
                listItem.addEventListener('click', function () {
                    searchInput.value = item;
                    resultContainer.innerHTML = ''; // Clear results when an item is selected
                    resultContainer.classList.add('hidden'); // Hide the results
\                });
                list.appendChild(listItem);
            });

            resultContainer.appendChild(list);
            resultContainer.classList.remove('hidden'); // Show the results

        } else {
            resultContainer.classList.add('hidden'); // Hide the results if no suggestions

        }
    }
});
