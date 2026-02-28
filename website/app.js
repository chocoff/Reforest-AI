const form = document.querySelector("#image-upload-form");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);

    try {
        // Send the image to the Flask backend
        const response = await fetch("http://localhost:5000/upload", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const blob = await response.blob(); // Get the binary image data
            const imageUrl = URL.createObjectURL(blob); // Create a temporary URL for the image

            // Display the converted image
            const imgElement = document.getElementById("converted-image");
            imgElement.src = imageUrl; // Update the image source
        } else {
            console.error("Failed to upload and process the image");
        }
    } catch (error) {
        console.error("Error:", error);
    }
});

    document.addEventListener('DOMContentLoaded', function() {
        const currentPath = window.location.pathname;

        // Find all navbar links
        const links = document.querySelectorAll('.navbar__links');
        links.forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    });
