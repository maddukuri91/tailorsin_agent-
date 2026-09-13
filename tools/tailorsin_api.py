# tools/tailorsin_api.py

import os
from typing import Optional

import requests


BASE_URL = "https://crm.tailorsin.com/tailorsin-api/api"


def client_registration(
    client_name: str,
    primary_no: str,
    secondary_no: Optional[str] = None
):
    """
    Register a new Tailorsin client.
    """

    url = f"{BASE_URL}/clientregistration.php"

    payload = {
        "client_name": client_name,
        "primary_no": primary_no,
        "secondary_no": secondary_no
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def bulk_order_enquiry(
    client_name: str,
    primary_no: str,
    secondary_no: Optional[str] = None
):
    """
    Submit a bulk order enquiry.
    """

    url = f"{BASE_URL}/bulkorderenquiry.php"

    payload = {
        "client_name": client_name,
        "primary_no": primary_no,
        "secondary_no": secondary_no
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def custom_fabric_estimation(
    client_name: str,
    primary_no: str,
    secondary_no: Optional[str] = None
):
    """
    Request custom fabric estimation.

    """

    url = f"{BASE_URL}/customfabricestimation.php"

    payload = {
        "client_name": client_name,
        "primary_no": primary_no,
        "secondary_no": secondary_no
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()



def get_client_type(mobile: str):
    """
    Classify an existing client by mobile number.

    Returns a dict with a 'type' field that is one of:
    'active_client', 'client' or 'new_user'.
    """
    url = f"{BASE_URL}/getclient.php"

    params = {
        "mobile": mobile
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def book_store_visit(
    mobile: str,
    store_id: int,
    bookdate: str,
    booktime: str
):
    """
    Book an appointment for a client.
    """

    url = f"{BASE_URL}/bookappointment.php"

    payload = {
        "mobile": mobile,
        "store_id": store_id,
        "bookdate": bookdate,
        "booktime": booktime
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_client_address(mobile: str):
    """
    Fetch the addresses of a client by mobile number.
    """

    url = f"{BASE_URL}/customeraddress.php"

    params = {
        "mobile": mobile
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()



def add_client_address(
    mobile: str,
    address: str,
    address2: Optional[str] = None,
    locality: Optional[str] = None,
    city: Optional[str] = None,
    pincode: Optional[str] = None
):
    """
    Add a new address for a client.
    """

    url = f"{BASE_URL}/addaddress.php"

    payload = {
        "mobile": mobile,
        "address": address,
        "address2": address2,
        "locality": locality,
        "city": city,
        "pincode": pincode
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def schedule_pickup(
    mobile: str,
    pickup_date: str,
    pickup_time: int,
    address_id: int
):
    """
    Schedule a fabric pickup for a client.
    """

    url = f"{BASE_URL}/schedulepickup.php"

    payload = {
        "mobile": mobile,
        "pickup_date": pickup_date,
        "pickup_time": pickup_time,
        "address_id": address_id
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()




def ship_fabric_to_store(
    mobile: str,
    store_id: int,
    notes: Optional[str] = None
):
    """
    Notify Tailorsin that a client is shipping fabric to a store for pickup.
    """

    url = f"{BASE_URL}/fabricdelivery.php"

    payload = {
        "mobile": mobile,
        "store_id": store_id,
        "notes": notes
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()



def human_handover(
    mobile: str
):
    """
    Request a human handover for a client.
    """

    url = f"{BASE_URL}/humanhandover.php"

    payload = {
        "mobile": mobile
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()

