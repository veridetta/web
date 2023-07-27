from django.shortcuts import render, redirect, get_object_or_404
from .forms import CreateUserForm, ContactForm, LoginForm, UpdateUserForm, DestinationForm
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
from . token import user_tokenizer_generate
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.models import auth
from django.contrib.auth import authenticate
from django.contrib.auth.decorators import login_required
from .models import Destination, Review
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.template.loader import render_to_string
from django.utils.html import strip_tags
import aiml
from django.http import JsonResponse
from .models import Destination
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.core.cache import cache
import string
import re #regex library
import os

import requests


# View halaman depan
def index(request):

    return render(request, 'index.html')


# Chatbot feature
def chatbot_response(request):
    message = request.GET.get('message', None)

    # Create the kernel and learn AIML files
    kernel = aiml.Kernel()
    kernel.learn('static/std-startup.xml')

    if message:
        bot_response = kernel.respond(message)
    else:
        bot_response = ''

    data = {
        'response': bot_response
    }

    return JsonResponse(data)


def about(request):

    return render(request, 'about.html')


# View registrasi
def register(request):

    form = CreateUserForm()
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.is_active = False
            user.save()

            # Email verification setup (template)
            current_site = get_current_site(request)
            subject = 'Account verification email'
            message = render_to_string('account/email-verification.html', {

                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': user_tokenizer_generate.make_token(user),

            })

            user.email_user(subject=subject, message=message)
            return redirect('email-verification-sent')

    context = {'form':form}
    return render(request, 'account/register.html', context=context)


def email_verification(request, uidb64, token):
    # uniqueid
    unique_id = force_str(urlsafe_base64_decode(uidb64))
    user = User.objects.get(pk=unique_id)
    # Success
    if user and user_tokenizer_generate.check_token(user, token):
        user.is_active = True
        user.save()
        return redirect('email-verification-success')
    # Failed
    else:
        return redirect('email-verification-failed')


def email_verification_sent(request):

    return render(request, 'account/email-verification-sent.html')


def email_verification_success(request):

    return render(request, 'account/email-verification-success.html')


def email_verification_failed(request):

    return render(request, 'account/email-verification-failed.html')


# Login view
def my_login(request):

    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)

        if form.is_valid():
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                auth.login(request, user)
                return redirect("index")

    context = {'form':form}
    return render(request, 'account/my-login.html', context=context)


# Logout view
def user_logout(request):
    try:
        for key in list(request.session.keys()):
            if key == 'session_key':
                continue
            else:
                del request.session[key]

    except KeyError:
        pass

    messages.success(request, "Logout success")
    return redirect("my-login")


# Dasboard view
@login_required(login_url='my-login')
def dashboard(reques):

    return render(reques, 'account/dashboard.html')


# Profile management
@login_required(login_url='my-login')
def profile_management(request):
    # Updating our user's username and email
    user_form = UpdateUserForm(instance=request.user)
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()
            messages.info(request, "Update success!")
            return redirect('dashboard')

    context = {'user_form':user_form}
    return render(request, 'account/profile-management.html', context=context)


# Delete account
@login_required(login_url='my-login')
def delete_account(request):
    user = User.objects.get(id=request.user.id)

    if request.method == 'POST':
        user.delete()
        messages.error(request, "Account deleted")
        return redirect('index')

    return render(request, 'account/delete-account.html')


# View for Contact Us
def contact(request):
    form = ContactForm(request.POST or None)
    if form.is_valid():
        full_name = form.cleaned_data.get("full_name")
        email = form.cleaned_data.get("email")
        message = form.cleaned_data.get("message")

        subject = "A new contact message"
        from_email = settings.EMAIL_HOST_USER
        to_email = [from_email]
        contact_message = "%s: %s via %s"%(full_name, message, email)

        send_mail(subject, contact_message, from_email, to_email, fail_silently=True)
        form = ContactForm()  # Reset form
        storage = messages.get_messages(request)
        # This will remove all messages
        for _ in storage:
            pass
        messages.success(request, 'Your message has been sent!')  # Add the success message

    context = {
        'form': form
    }
    return render(request, 'contact.html', context)


# Get weather data
def get_weather_data(lat, lon, api_key):
    response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}")
    return response.json()

# Detail page function
def destination_detail(request, destination_id):
    # Mengambil objek Destination berdasarkan id yang diberikan. Jika tidak ada, akan menghasilkan error 404.
    destination = get_object_or_404(Destination, id=destination_id)
    title = destination.title
    # Mengambil semua objek Review yang memiliki destination yang sama dengan objek destination yang telah diambil sebelumnya.
    reviews = Review.objects.filter(destination=destination)

    # Jika metode request adalah POST (biasanya dari form submit), maka akan membuat review baru.
    if request.method == 'POST':
        review_text = request.POST.get('review_text')
        new_review = Review(user=request.user, destination=destination, review_text=review_text)
        new_review.save()


    import pandas as pd
    import numpy as np
    
    place_data = get_preprocessed_data_from_cache_or_db()
    place_data.columns = ["id", "title", "image","description","","","","","","","","","place_tokens_stemmed"]

    place_data.head()
    print('head')
    print(place_data.head(5));
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    #//SINI
    # Simpan hasil pemrosesan teks pada DataFrame temporari
    processed_data = place_data[["id", "title", "image","description","","","","","","","","","place_tokens_stemmed"]].copy()

    # Ubah kolom place_tokens_stemmed menjadi list of strings
    processed_data['place_tokens_stemmed'] = processed_data['place_tokens_stemmed'].apply(lambda x: ' '.join(x))

    # Buat TF-IDF Vectorizer
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                        token_pattern=r'\w+', ngram_range=(1, 10), stop_words='english')
    tfv = TfidfVectorizer (min_df=3, max_features=None,
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w+',
                        ngram_range=(1, 10),
                        stop_words = 'english')
    # Fit dan transform data untuk mendapatkan matrix tf-idf
    tfv_matrix = tfv.fit_transform(processed_data['place_tokens_stemmed'])

    # Hitung cosine similarity
    cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

    # Buat dictionary untuk mapping antara nama tempat dan indeksnya pada DataFrame
    indices = pd.Series(processed_data.index, index=processed_data['title']).drop_duplicates()
    #SINI
    
    #place_data['place_tokens_stemmed'] = place_data['place_tokens_stemmed'].fillna('')
    #tfv_matrix = tfv.fit_transform(place_data['place_tokens_stemmed'])

    #cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

    #indices = pd.Series(place_data.index, index=place_data['title']).drop_duplicates()

    list(enumerate(cosine_sim[indices['Tahu Petis Kertasari']]))

    sorted(list(enumerate(cosine_sim[indices['Tahu Petis Kertasari']])), key=lambda x: x[1], reverse=True)

    idx = indices[title]
    cosine_sim_scores = list(enumerate(cosine_sim[idx]))
    cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)
    cosine_sim_scores = cosine_sim_scores[1:6]
    place_data_indices = [i[0] for i in cosine_sim_scores]

    # Mendapatkan data rekomendasi berdasarkan indeks
    related_destinations = place_data[['id','title', 'description','image']].iloc[place_data_indices[:4]]
    ids = related_destinations['id'].tolist()
    titles = related_destinations['title'].tolist()
    descriptions = related_destinations['description'].tolist()
    images = related_destinations['image'].tolist()
    # Menggabungkan keempat list menjadi satu list
    combined_list = list(zip(ids, titles, descriptions, images))
    print("related")
    print(images)
    print(related_destinations)
    # Mengambil data cuaca.
    weather_data = get_weather_data(destination.latitude, destination.longitude, "5aa3cbbe4f08e84023a8e2b4cb638ba0")
    # Mengubah suhu dari Kelvin ke Celsius jika data suhu tersedia dalam data cuaca.
    if 'main' in weather_data and 'temp' in weather_data['main']:
        weather_data['main']['temp'] = weather_data['main']['temp'] - 273.15

    # Me-render template 'destination_detail.html' dengan konteks yang berisi objek destination, objek reviews, objek related_destinations, dan data cuaca.
    return render(request, 'destination_detail.html', {
    'destination': destination,
    'relats':combined_list,
    'reviews': reviews,
    'ids': ids,
    'titles': titles,
    'descriptions': descriptions,
    'images': images,
    })


def get_preprocessed_data_from_cache_or_db():
    cached_data = cache.get('preprocessed_data')

    if not cached_data:
        # Jika data tidak ada di cache, ambil data dari database dan lakukan preprocessing
        import pandas as pd
        import numpy as np
        import os

        #csv_place_data = os.path.join(settings.BASE_DIR, 'static', 'dataset (1).csv')
        #place_data = pd.read_csv(csv_place_data)
        # Ambil data dari tabel myapp_destination menggunakan Django ORM
        place_data = pd.DataFrame(list(Destination.objects.all().values()))
        # Case folding
        place_data['description'] = place_data['description'].str.lower()
        print('Case folding result : \n')
        print(place_data['description'].head(5))
        print('\n\n\n')
        import string
        import re #regex library
        import nltk
        #import word_tokenize & FreqDist from NLTK
        from nltk.tokenize import word_tokenize
        from nltk.probability import FreqDist
        nltk.download('punkt')
        #Tokenizing

        def remove_place_special(text):
            if isinstance(text, str):
                #remove tab, new line, and back slice
                text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ")
                #remove non ASCII (emoticon, chinese word, etc)
                text = text.encode('ascii', 'replace').decode('ascii')
                #remove mention, link, hastag
                text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
                #remove incomplete url
                return text.replace("http://", " ").replace("https://", " ")

        place_data['description'] = place_data['description'].apply(remove_place_special)

        #remove number
        def remove_number(text):
            if isinstance(text, str):
                return re.sub(r"\d+", "", text)

        place_data['description'] = place_data['description'].apply(remove_number)

        #remove punctuation
        def remove_punctuation(text):
            if isinstance(text, str):
                return text.translate(str.maketrans("","",string.punctuation))

        place_data['description'] = place_data['description'].apply(remove_punctuation)

        #remove whitespace leading & trailing
        def remove_whitespace_LT(text):
            if isinstance(text, str):
                return text.strip()

        place_data['description'] = place_data['description'].apply(remove_whitespace_LT)

        #remove multiple whitespace into single whitespace
        def remove_whitespace_multiple(text):
            if isinstance(text, str):
                return re.sub('\s+',' ',text)

        place_data['description'] = place_data['description'].apply(remove_whitespace_multiple)

        #remove single char
        def remove_single_char(text):
            if isinstance(text, str):
                return re.sub(r"\b[a-zA-Z]\b", "", text)

        place_data['description'] = place_data['description'].apply(remove_single_char)

        #nltk word tokenize
        def word_tokenize_wrapper(text):
            if isinstance(text, str):
                return word_tokenize(text)

        place_data['place_tokens'] = place_data['description'].apply(word_tokenize_wrapper)

        print('Tokenixing Result : \n')
        print(place_data['place_tokens'].head())
        print('\n\n\n')

        #nltk calc frequency distribution
        def freqDist_wrapper(text):
            return FreqDist(text)

        place_data['place_tokens_fdist'] = place_data['place_tokens'].apply(freqDist_wrapper)

        print('Frequency tokens : \n')
        print(place_data['place_tokens_fdist'].head().apply(lambda x : x.most_common()))

        from nltk.corpus import stopwords
        nltk.download("stopwords")
        #get stopword from nltk stopword
        #get stopword indonesia
        list_stopwords = stopwords.words('indonesian')

        #convert list to dictionary
        list_stopwords = set(list_stopwords)

        #remove stopword pada list token
        def stopwords_removal(words):
            if words is not None:
                return [word for word in words if word not in list_stopwords]
            else:
                return []

        place_data['place_tokens_WSW'] = place_data['place_tokens'].apply(stopwords_removal)

        print(place_data['place_tokens_WSW'].head())

        #import sastrawi package
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        import swifter

        #create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        #stemmed
        def stemmed_wrapper(term):
            return stemmer.stem(term)

        term_dict = {}

        for document in place_data['place_tokens_WSW']:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '

        print(len(term_dict))
        print("-------------------")

        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])

        print(term_dict)
        print("-------------------")

        #apply stemmed term to dataframe
        def get_stemmed_term(document):
            return [term_dict[term] for term in document]
        print(place_data.columns)

        place_data['place_tokens_stemmed'] = place_data['place_tokens_WSW'].swifter.apply(get_stemmed_term)
        print(place_data['place_tokens_stemmed'])
        # Setelah selesai preprocessing, simpan data yang telah diolah di cache selama 5 menit (300 detik)
        place_data.to_csv("static/hasil_processing.csv")
        cache.set('preprocessed_data', place_data, timeout=300)
    else:
        # Jika data ada di cache, gunakan data yang ada
        place_data = cached_data

    return place_data
@login_required
def add_review(request, destination_id):
    if request.method == 'POST':
        review_text = request.POST['review_text']
        destination = get_object_or_404(Destination, pk=destination_id)
        Review.objects.create(user=request.user, destination=destination, review_text=review_text)
    return redirect('destination_detail', destination_id=destination_id)

def search_view(request):
    if request.method == 'GET':
        category = request.GET.get('category')
        budget = request.GET.get('budget')

        # Ambil data destinasi dari database sesuai dengan kategori dan budget yang dipilih
        destinations = Destination.objects.filter(category=category, budget=budget)

        # Jika tidak ada hasil pencarian
        if not destinations.exists():
            message = "No destinations found for the selected category and budget."
            return render(request, 'search.html', {'message': message})

        # Proses Content-Based Filtering (CBF)
        tfv = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfv.fit_transform(destinations.values_list('description', flat=True))
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Ambil daftar judul destinasi untuk rekomendasi
        destination_titles = destinations.values_list('title', flat=True)
        destination_indices = dict(zip(destination_titles, range(len(destination_titles))))

        # Tentukan destinasi untuk direkomendasikan berdasarkan similarity score tertinggi
        recommendations = []
        for title in destination_titles:
            idx = destination_indices[title]
            cosine_sim_scores = list(enumerate(cosine_sim[idx]))
            cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)
            cosine_sim_scores = cosine_sim_scores[1:6]  # Ambil 5 destinasi dengan similarity score tertinggi
            recommended_destinations = [destination_titles[i[0]] for i in cosine_sim_scores]
            recommendations.append({'title': title, 'recommendations': recommended_destinations})

        return render(request, 'search.html', {'destinations': recommendations})

    return render(request, 'index.html')
# Search function
def search(request):
    if request.method == 'GET':
        category = request.GET.get('category', '')
        budget = request.GET.get('budget', '')

        # Jika kategori dan anggaran kosong, berarti user belum melakukan pencarian
        if not category and not budget:
            return render(request, 'index.html')

        # Filter destinasi berdasarkan kategori dan anggaran yang dipilih oleh user
        destinations = Destination.objects.filter(category=category, budget=budget)

        # Jika tidak ada hasil sesuai kriteria, berikan rekomendasi berdasarkan CBF
        if not destinations:
            # Ambil indeks destinasi yang sesuai dari hasil cosine_similarity
            idx = indices[category]  # Misalnya, gunakan kategori sebagai kunci untuk mengambil indeksnya
            cosine_sim_scores = list(enumerate(cosine_sim[idx]))
            cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)
            cosine_sim_scores = cosine_sim_scores[1:6]
            place_data_indices = [i[0] for i in cosine_sim_scores]

            # Ambil destinasi berdasarkan indeks yang didapat
            recommended_destinations = Destination.objects.filter(id__in=place_data.iloc[place_data_indices]['id'])

            return render(request, 'search.html', {'destinations': recommended_destinations})
        else:
            return render(request, 'search.html', {'destinations': destinations})


# Create a new destination using an HTML form called create_destination
# Only admin can access this facility
def admin_check(user):
    return user.is_staff or user.is_superuser


@login_required
@user_passes_test(admin_check)
def create_destination(request):
    if request.method == 'POST':
        form = DestinationForm(request.POST, request.FILES)
        if form.is_valid():
            new_destination = form.save()
            messages.success(request, 'Destinasi berhasil dibuat.')

            # Get all users email
            recipients = [user.email for user in User.objects.all()]

            # Prepare email content
            subject = f"New Destination Created: {new_destination.title}"
            html_message = render_to_string('email_notification.html', {'destination': new_destination})
            plain_message = strip_tags(html_message)

            # Send email to all users
            send_mail(
                subject,  # subject
                plain_message,  # message
                'nextkoding@gmail.com',  # from email
                recipients,  # recipient list
                html_message=html_message,
                fail_silently=False,
            )

            return redirect('destination_detail', destination_id=new_destination.id)

    else:
        form = DestinationForm()

    context = {'form': form}
    return render(request, 'create_destination.html', context)


# Update & delete function - only admin
def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin, login_url='my-login')
def update_destination(request, id):
    destination = get_object_or_404(Destination, id=id)
    if request.method == "POST":
        form = DestinationForm(request.POST, instance=destination)
        if form.is_valid():
            form.save()
            messages.success(request, "Destination updated successfully!")
            return redirect('destination_detail', destination_id=destination.id)
    else:
        form = DestinationForm(instance=destination)
    return render(request, 'update_destination.html', {'form': form, 'destination': destination})


@login_required
@user_passes_test(is_admin, login_url='my-login')
def delete_destination(request, id):
    destination = get_object_or_404(Destination, id=id)
    if request.method == 'POST':
        destination.delete()
        messages.success(request, "Destination deleted successfully!")
        return redirect('index')
    return render(request, 'delete_destination.html', {'destination': destination})