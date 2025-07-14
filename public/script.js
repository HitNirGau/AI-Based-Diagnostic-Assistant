document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed.");  // Check if DOM is loaded

    fetch('../data/patients.csv')
        .then(response => response.text())
        .then(data => {
            console.log("CSV Data Loaded Successfully");  // Confirm CSV load
            const patientData = parseCSV(data);
            console.log('adding patient data');
            try {
                displayPatients(patientData);
                console.log('Added');  // This will only show if no error occurs
            } catch (error) {
                console.error("Error in displayPatients:", error);  // Catch errors here
            }
            // const searchButton = document.getElementById('searchSimilarImages');
            // if (searchButton) {
            //     searchButton.style.display = 'inline-block';
            // } else {
            //     console.warn("Element with id 'searchSimilarImages' not found.");
            // }
        })
        .catch(error => console.error("Error loading CSV file:", error));
});



// Correct CSV Parsing Function

function parseCSV(data) {

    const lines = data.trim().split('\n');

    const headers = lines[0].split(',').map(header => header.trim());  // Trim headers



    return lines.slice(1).map(line => {

        const values = line.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g) || []; // Handles commas inside quotes

        const patient = {};



       headers.forEach((header, index) => {

            patient[header] = values[index]?.replace(/(^"|"$)/g, '').trim() || ''; // Remove surrounding quotes

        });



        return patient;

    });

}



// Display patients in the table

function displayPatients(patients) {

    const tbody = document.getElementById('patientTableBody'); // Correct ID used here
    if (!tbody) {
        console.error("Element with id 'patientTableBody' not found.");
        return;
    }
    tbody.innerHTML = '';
    patients.forEach(patient => {

        const row = `

            <tr>

                <td>${patient["Doctor's Name"]}</td>


                <td>${patient["Patient Name"]}</td>

                <td>${patient.Age}</td>

                <td>${patient.Gender}</td>

                <td>${patient.Contact}</td>

                <td>${patient["Medical History"]}</td>

                <td>${patient.Medications}</td>

                <td>${patient.Allergies}</td>

                <td>${patient.Diagnosis}</td>

                <td>${patient.Date}</td>

            </tr>

        `;

        tbody.innerHTML += row;
        tbody.style.display = 'none';  // Hide temporarily
        tbody.offsetHeight;            // Trigger reflow
        tbody.style.display = 'table-row-group';  // Show again


    });
   console.log("displayed");
}



// Search functionality

function searchPatients() {
    console.log("searching");  // Fixed typo from "searcing" to "searching"

    const query = document.getElementById('searchInput').value.toLowerCase();

    fetch(`/history/patients/search?name=${query}`)
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            console.log("Data received:", data);

            // Fixed logical error: Use `||` instead of a comma to combine conditions
            const filtered = data.filter(patient => 
                (patient['Patient Name'] && patient['Patient Name'].toLowerCase().includes(query)) ||
                (patient['Doctor Name'] && patient['Doctor Name'].toLowerCase().includes(query))
            );

            populatePatientTable(filtered);
        })
        .catch(error => console.error("Search Error:", error));
}



// Fetch patients from server

function fetchPatients() {

    fetch('/history/patients')

        .then(response => response.json())

        .then(patients => populatePatientTable(patients))

        .catch(error => console.error('Error fetching patients:', error));

}



// Populate the patient table

function populatePatientTable(patients) {

    const tableBody = document.getElementById('patientTableBody');

    tableBody.innerHTML = '';



    patients.forEach(patient => {

        const row = document.createElement('tr');

        row.innerHTML = `

            <td>${patient.doctorName}</td>

            <td class="text-primary patient-name" style="cursor: pointer;">${patient.patientName}</td>

            <td>${patient.age}</td>

            <td>${patient.gender}</td>

            <td>${patient.contact}</td>

            <td>${patient.medicalHistory}</td>

            <td>${patient.medications}</td>
            
            <td>${patient.allergies}</td>

            <td>${patient.diagnosis}</td>
            <td>${patient.date}</td>

        `;



        row.querySelector('.patient-name').addEventListener('click', () => showPatientReport(patient));



        tableBody.appendChild(row);

    });

}



// Function to display patient report

function showPatientReport(patient) {

    const reportDiv = document.getElementById('patientReport');

    reportDiv.style.display = 'block';

    reportDiv.innerHTML = `

        <div class="card shadow p-4 bg-light rounded">

            <h3 class="text-center text-primary">Patient Report</h3>

            <p><strong>Doctor's Name:</strong> ${patient.doctorName}</p>

            <p><strong>Patient Name:</strong> ${patient.patientName}</p>

            <p><strong>Age:</strong> ${patient.age}</p>

            <p><strong>Gender:</strong> ${patient.gender}</p>

            <p><strong>Contact:</strong> ${patient.contact}</p>

            <p><strong>Medical History:</strong> ${patient.medicalHistory}</p>

            <p><strong>Medications:</strong> ${patient.medications}</p>

            <p><strong>Allergies:</strong> ${patient.allergies}</p>

            <p><strong>Diagnosis:</strong> ${patient.diagnosis}</p>

            <p><strong>Date:</strong> ${patient.date}</p>

        </div>

    `;

}
