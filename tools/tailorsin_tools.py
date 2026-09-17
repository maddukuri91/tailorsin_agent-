# tools/tailorsin_tools.py

from langchain_core.tools import tool

from tools.tailorsin_api import (
    client_registration,
    bulk_order_enquiry,
    custom_fabric_estimation,
    get_client_type,
    book_store_visit as api_book_store_visit,
    get_client_address as api_get_client_address,
    add_client_address as api_add_client_address,
    schedule_pickup as api_schedule_pickup,
    ship_fabric_to_store as api_ship_fabric_to_store,
    human_handover as api_human_handover,
    get_order_status as api_get_order_status,
    cancel_order as api_cancel_order,
    modify_order as api_modify_order,
)


@tool
def register_client(
    client_name: str,
    primary_no: str,
    secondary_no: str = ""
):
    """
    Register a customer in the Tailorsin CRM.

    Use this only when the required customer information
    has been collected.
    """

    return client_registration(
        client_name=client_name,
        primary_no=primary_no,
        secondary_no=secondary_no or None
    )


@tool
def submit_bulk_order_enquiry(
    client_name: str,
    primary_no: str,
    secondary_no: str = ""
):
    """
    Submit a bulk order enquiry to Tailorsin CRM.
    """

    return bulk_order_enquiry(
        client_name=client_name,
        primary_no=primary_no,
        secondary_no=secondary_no or None
    )


@tool
def get_custom_fabric_estimation(
    client_name: str,
    primary_no: str,
    secondary_no: str = ""
):
    """
    Request a custom fabric estimation.

    Sends the customer's name and contact details to the CRM so the
    Tailorsin team can prepare the estimation.

    Required information:
    - client_name
    - primary_no (the customer's contact number)
    - secondary_no (optional — only if a separate number was given)
    """

    return custom_fabric_estimation(
        client_name=client_name,
        primary_no=primary_no,
        secondary_no=secondary_no or None
    )


@tool
def classify_client(mobile: str):
    """
    Classify a customer by their mobile number.

    Returns the customer type which is one of:
    - active_client
    - client
    - new_user
    """

    return get_client_type(mobile=mobile)


@tool
def book_store_visit(
    mobile: str,
    store_id: int,
    bookdate: str,
    booktime: str
):
    """
    Book a store visit for a customer.

    - store_id: which Tailorsin store
    - bookdate: e.g. '2026-07-10'
    - booktime: e.g. '11:00 AM - 12:00 PM'
    """

    return api_book_store_visit(
        mobile=mobile,
        store_id=store_id,
        bookdate=bookdate,
        booktime=booktime
    )


@tool
def get_client_address(mobile: str):
    """
    Fetch the saved addresses for a customer.
    """

    return api_get_client_address(mobile=mobile)


@tool
def add_client_address(
    mobile: str,
    address: str,
    address2: str = "",
    locality: str = "",
    city: str = "",
    pincode: str = ""
):
    """
    Add a new address for a customer.

    Required:
    - mobile
    - address (first address line)

    Optional:
    - address2, locality, city, pincode
    """

    return api_add_client_address(
        mobile=mobile,
        address=address,
        address2=address2 or None,
        locality=locality or None,
        city=city or None,
        pincode=pincode or None
    )


@tool
def schedule_pickup(
    mobile: str,
    pickup_date: str,
    pickup_time: int,
    address_id: int
):
    """
    Schedule a fabric pickup for a customer.

    - pickup_date: e.g. '2026-07-10'
    - pickup_time: a time-slot id (integer)
    - address_id: the id of the customer's saved pickup address
    """

    return api_schedule_pickup(
        mobile=mobile,
        pickup_date=pickup_date,
        pickup_time=pickup_time,
        address_id=address_id
    )


@tool
def ship_fabric_to_store(
    mobile: str,
    store_id: int,
    notes: str = ""
):
    """
    Notify Tailorsin that a customer is shipping fabric to a store for pickup.
    """

    return api_ship_fabric_to_store(
        mobile=mobile,
        store_id=store_id,
        notes=notes or None
    )


@tool
def human_handover(mobile: str):
    """
    Request a human handover for a customer.
    """

    return api_human_handover(mobile=mobile)


@tool
def get_order_status(mobile: str):
    """
    Get the current status of an order for a customer.

    - mobile: the customer's mobile number

    Returns the order status from the CRM.
    """

    return api_get_order_status(mobile=mobile)


@tool
def cancel_order(mobile: str, order_id: int, reason: str):
    """
    Cancel an existing order for a customer.

    - mobile: the customer's mobile number
    - order_id: the ID of the order to cancel
    - reason: the reason for cancellation (e.g. 'Changed my mind')

    Returns the cancellation result from the CRM.
    """

    return api_cancel_order(
        mobile=mobile,
        order_id=order_id,
        reason=reason
    )


@tool
def modify_order(mobile: str, order_id: int, comment: str):
    """
    Modify an existing order for a customer.

    - mobile: the customer's mobile number
    - order_id: the ID of the order to modify
    - comment: the modification request (e.g. 'Please make the sleeves half instead of full')

    Returns the modification result from the CRM.
    """

    return api_modify_order(
        mobile=mobile,
        order_id=order_id,
        comment=comment
    )
