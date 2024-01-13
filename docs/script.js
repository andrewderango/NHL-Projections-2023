let teamIndex;

function loadCSV(file) {
		const table = document.getElementById('csvTable');
		const teamFilter = document.getElementById('teamFilter');
		table.innerHTML = '';
		teamFilter.innerHTML = '<option value="">All Teams</option>';

		fetch(file)
				.then(response => response.text())
				.then(data => {
						const rows = data.trim().split('\n');
						const header = rows.shift().split(',');

						// Find team column index
						teamIndex = header.indexOf('Team');
						if (teamIndex === -1) return; // Exit if there is no 'Team' column

						// Create a set of teams
						const teams = new Set();
						rows.forEach(row => {
								const columns = row.split(',');
								teams.add(columns[teamIndex]);
						});

						// Convert the set to an array and sort it
						const sortedTeams = Array.from(teams).sort();

						// Add teams to the dropdown list
						sortedTeams.forEach(team => {
								const option = document.createElement('option');
								option.value = option.textContent = team;
								teamFilter.appendChild(option);
						});

						// Create table header
						const headerRow = table.createTHead().insertRow();
						header.forEach(column => {
								const th = document.createElement('th');
								th.textContent = column;
								headerRow.appendChild(th);
						});

						// Create table rows
						rows.forEach(row => {
								const columns = row.split(',');
								const newRow = table.insertRow();
								columns.forEach((column, index) => {
										const cell = newRow.insertCell();
										cell.textContent = column;
								});
						});
				})
				.catch(error => console.error('Error:', error));
}

function searchTable() {
		const input = document.getElementById('searchInput');
		const teamFilter = document.getElementById('teamFilter');
		const filter = input.value.toUpperCase();
		const team = teamFilter.value.toUpperCase();
		const table = document.getElementById('csvTable');
		const trs = table.getElementsByTagName('tr');

		for (let i = 1; i < trs.length; i++) {
				let found = false;
				const tds = trs[i].getElementsByTagName('td');
				for (let j = 0; j < tds.length; j++) {
						if (tds[j].textContent.toUpperCase().indexOf(filter) > -1 && 
								(team === '' || tds[teamIndex].textContent.toUpperCase() === team)) {
								found = true;
								break;
						}
				}
				trs[i].style.display = found ? '' : 'none';
		}
}

loadCSV('projections.csv');
