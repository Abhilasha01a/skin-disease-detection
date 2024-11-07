document.getElementById('uploadButton').addEventListener('click', () => {
    document.getElementById('upload').click();
});

document.getElementById('upload').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            document.getElementById('video').style.display = 'none';
            document.getElementById('captureButton').style.display = 'none';
            document.getElementById('submitButton').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('cameraButton').addEventListener('click', () => {
    const video = document.getElementById('video');
    const captureButton = document.getElementById('captureButton');
    video.style.display = 'block';
    captureButton.style.display = 'block';
    document.getElementById('previewImage').style.display = 'none';
    document.getElementById('submitButton').style.display = 'none';

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            video.srcObject = stream;
            video.play();
        }).catch((err) => {
            console.error('Error accessing camera: ', err);
        });
    } else {
        alert('Camera not supported.');
    }
});

document.getElementById('captureButton').addEventListener('click', () => {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/png');
    document.getElementById('previewImage').src = dataURL;
    document.getElementById('previewImage').style.display = 'block';
    video.style.display = 'none';
    document.getElementById('captureButton').style.display = 'none';
    document.getElementById('submitButton').style.display = 'block';

    // Stop the video stream
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
});

document.getElementById('submitButton').addEventListener('click', () => {
    const previewImage = document.getElementById('previewImage');
    if (previewImage.src) {
        // Store the image URL in localStorage
        localStorage.setItem('imageUrl', previewImage.src);
        // Redirect to results.html
        window.location.href = 'results.html';
    } else {
        alert('Please upload or capture an image before submitting.');
    }
});
