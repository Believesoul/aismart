/* $Id: campaign_difficulty.cpp 49602 2011-05-22 17:56:13Z mordante $ */
/*
   Copyright (C) 2010 - 2011 by Ignacio Riquelme Morelle <shadowm2006@gmail.com>
   Part of the Battle for Wesnoth Project http://www.wesnoth.org/

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY.

   See the COPYING file for more details.
*/

#define GETTEXT_DOMAIN "rose-lib"

#include "gui/dialogs/combo_box.hpp"

#include "formula_string_utils.hpp"
#include "gettext.hpp"

#include "gui/dialogs/helper.hpp"
#include "gui/widgets/listbox.hpp"
#include "gui/widgets/settings.hpp"
#include "gui/widgets/window.hpp"
#include "gui/widgets/toggle_panel.hpp"
#include "gui/widgets/label.hpp"

#include <boost/bind.hpp>

namespace gui2 {

/*WIKI
 * @page = GUIWindowDefinitionWML
 * @order = 2_campaign_difficulty
 *
 * == Campaign difficulty ==
 *
 * The campaign mode difficulty menu.
 *
 * @begin{table}{dialog_widgets}
 *
 * title & & label & m &
 *         Dialog title label. $
 *
 * message & & scroll_label & o &
 *         Text label displaying a description or instructions. $
 *
 * listbox & & listbox & m &
 *         Listbox displaying user choices, defined by WML for each campaign. $
 *
 * -icon & & control & m &
 *         Widget which shows a listbox item icon, first item markup column. $
 *
 * -label & & control & m &
 *         Widget which shows a listbox item label, second item markup column. $
 *
 * -description & & control & m &
 *         Widget which shows a listbox item description, third item markup column. $
 *
 * @end{table}
 */

REGISTER_DIALOG(rose, combo_box)

tcombo_box::tcombo_box(const std::vector<tval_str>& items, int val, const std::string& title, bool allow_cancel)
	: index_(twidget::npos)
	, items_(items)
	, title_(title)
	, allow_cancel_(allow_cancel)
{
	VALIDATE(!items.empty(), null_str);

	for (std::vector<tval_str>::const_iterator it = items.begin(); it != items.end(); ++ it) {
		if (it->val == val) {
			index_ = std::distance(items.begin(), it);
			break;
		}
	}
}

void tcombo_box::pre_show(CVideo& /*video*/, twindow& window)
{
	window.set_canvas_variable("border", variant("default-border"));

	tlabel* title = find_widget<tlabel>(&window, "title", false, true);
	if (!title_.empty()) {
		title->set_label(title_);
	} else {
		title->set_visible(twidget::INVISIBLE);
	}
	if (!allow_cancel_) {
		find_widget<twidget>(&window, "ok", false, true)->set_visible(twidget::INVISIBLE);
		find_widget<twidget>(&window, "cancel", false, true)->set_visible(twidget::INVISIBLE);
	}
	tlistbox& list = find_widget<tlistbox>(&window, "listbox", false);
	window.keyboard_capture(&list);

	std::map<std::string, std::string> data;

	int item_index = 0;
	for (std::vector<tval_str>::const_iterator it = items_.begin(); it != items_.end(); ++ it) {
		const tval_str& item = *it;
		
		data["label"] = item.str;
		list.insert_row(data);

		item_index ++;
	}

	if (index_ != twidget::npos) {
		list.select_row(index_);
	}
	list.set_did_row_changed(boost::bind(&tcombo_box::item_selected, this, boost::ref(window), _1, _2));
}

void tcombo_box::item_selected(twindow& window, tlistbox& list, ttoggle_panel& widget)
{
	if (!allow_cancel_) {
		window.set_retval(twindow::OK);
	}
}

void tcombo_box::post_show(twindow& window)
{
	tlistbox& list = find_widget<tlistbox>(&window, "listbox", false);
	ttoggle_panel* selected = list.cursel();
	index_ = selected? selected->at(): twidget::npos;
}

}
