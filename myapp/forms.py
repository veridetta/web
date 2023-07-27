from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.forms.widgets import PasswordInput, TextInput
from django.contrib.auth.models import User
from django import forms
from .models import Destination


# Registration form
class CreateUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

# Tentukan kelas bernama CreateUserForm dengan *args dan **kwargs sebagai parameter untuk konstruktor
    def __init__(self, *args, **kwargs):
        # Panggil konstruktor kelas induk 'super' dengan meneruskan argumen yang diwariskan
        super(CreateUserForm, self).__init__(*args, **kwargs)
        #  Form 'email'menjadi wajib diisi saat formulir disubmit
        self.fields['email'].required = True

    # Fungsi ini digunakan untuk memvalidasi input email
    def clean_email(self):
        email = self.cleaned_data.get("email")
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('This email is invalid')
        # len function updated ###
        if len(email) >= 350:
            raise forms.ValidationError("Your email is too long")

        return email


# Contact us form
class ContactForm(forms.Form):
    full_name = forms.CharField(required=True)
    email = forms.EmailField(required=True)
    message = forms.CharField(widget=forms.Textarea, required=True)


# Login form
class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=TextInput())
    password = forms.CharField(widget=PasswordInput())


# Update form
class UpdateUserForm(forms.ModelForm):
    password = None
    class Meta:
        model = User
        fields = ['username', 'email']
        exclude = ['password1', 'password1']

    def __init__(self, *args, **kwargs):
        super(UpdateUserForm, self).__init__(*args, **kwargs)
        # Mark email as required
        self.fields['email'].required = True

    # Email validation
    def clean_email(self):
        email = self.cleaned_data.get("email")
        if User.objects.filter(email=email).exclude(pk=self.instance.pk).exists():
            raise forms.ValidationError('This email is invalid')
        # len function updated ###
        if len(email) >= 350:
            raise forms.ValidationError("Your email is too long")

        return email


# Destination form
class DestinationForm(forms.ModelForm):
    class Meta:
        model = Destination
        fields = ['title', 'image', 'description', 'category', 'budget', 'latitude', 'longitude', 'youtube_url']
