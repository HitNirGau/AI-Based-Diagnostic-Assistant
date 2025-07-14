function myFunction() {
  document.getElementById("myDropdown").classList.toggle("show");
}

// Close the dropdown menu if the user clicks anywhere outside
window.onclick = function (event) {
  if (!event.target.matches('.dropbtn')) {
    var dropdowns = document.getElementsByClassName("dropdown-content");
    var i;
    for (i = 0; i < dropdowns.length; i++) {
      var openDropdown = dropdowns[i];
      if (openDropdown.classList.contains('show')) {
        openDropdown.classList.remove('show');
      }
    }
  }

}

function openSidebar() {
  document.getElementById("sidePanel").style.left = "0";
}

function closeSidebar() {
  document.getElementById("sidePanel").style.left = "-250px";
}

function redirectToHistory() {
  window.location.href = "/dash/patienthistory";
}

function redirectToAddPatient() {
  window.location.href = "/dash/getadd";
}

function redirectToUploadXray() {
  window.location.href = '/dash/upload_xray';
}
